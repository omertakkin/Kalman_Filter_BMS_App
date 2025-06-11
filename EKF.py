import numpy as np

class EKF:
    def __init__(self, v0, T0, SigmaX0, SigmaV, SigmaW, model):
        
        # Store model
        self.model = model

        # Initialize state description
        ir0 = 0.0  # Initial diffusion current
        hk0 = 0.0  # Initial hysteresis voltage
        SOC0 = self.SOCfromOCVtemp(v0, T0)

        # State variable indices
        self.irInd = 0
        self.hkInd = 1
        self.zkInd = 2

        # Initial state (column matrix)
        self.xhat = np.array([[SOC0], [ir0], [hk0]])

        # Covariances
        self.SigmaW = SigmaW
        self.SigmaV = SigmaV
        self.SigmaX = SigmaX0
        self.SXbump = 5

        # Previous current value
        self.priorI = 0.0
        self.signIK = 0.0


    def SOCfromOCVtemp(self, v, T):
        """
        Interpolate the state of charge (SOC) from open-circuit voltage v0 and temperature T0.
        Args:
            v: voltage (V)
            T: temperature (°C)
        Returns:
            SOC value between 0 and 1
        """
        soc = np.array(self.model['SOC0'].dropna())
        ocv0 = np.array(self.model['OCV0'].dropna())
        ocvrel = np.array(self.model['OCVrel'].dropna())

        # Ensure all arrays are aligned in length
        min_len = min(len(soc), len(ocv0), len(ocvrel))
        soc, ocv0, ocvrel = soc[:min_len], ocv0[:min_len], ocvrel[:min_len]

        # Calculate OCV at given temperature
        ocv_temp = ocv0 + T * ocvrel

        # Sort arrays based on OCV for proper interpolation
        sort_idx = np.argsort(ocv_temp)
        ocv_sorted = ocv_temp[sort_idx]
        soc_sorted = soc[sort_idx]

        # Handle out-of-range values
        if v <= ocv_sorted[0]:
            print(f"Warning: Voltage {v:.2f}V below minimum OCV {ocv_sorted[0]:.2f}V. Clamping SOC to {soc_sorted[0]:.2%}")
            return float(soc_sorted[0])
        if v >= ocv_sorted[-1]:
            print(f"Warning: Voltage {v:.2f}V above maximum OCV {ocv_sorted[-1]:.2f}V. Clamping SOC to {soc_sorted[-1]:.2%}")
            return float(soc_sorted[-1])

        # Interpolate SOC value
        return float(np.interp(v, ocv_sorted, soc_sorted))
    
    def OCVfromSOCtemp(self, z, T):
        """
        Compute the open-circuit voltage for a given state of charge z (0–1) and temperature T (°C).
        Args:
            z: SOC value between 0 and 1
            T: temperature (°C)
        Returns:
            OCV value in volts
        """
        # Input validation
        if not 0 <= z <= 1:
            print(f"Warning: SOC {z:.2%} outside valid range [0,1]. Clamping to valid range.")
            z = np.clip(z, 0, 1)

        soc = np.array(self.model['SOC0'].dropna())
        ocv0 = np.array(self.model['OCV0'].dropna())
        ocvrel = np.array(self.model['OCVrel'].dropna())

        # Ensure all arrays are aligned in length
        min_len = min(len(soc), len(ocv0), len(ocvrel))
        soc, ocv0, ocvrel = soc[:min_len], ocv0[:min_len], ocvrel[:min_len]

        # Calculate OCV components
        ocv_base = float(np.interp(z, soc, ocv0))
        ocv_temp = float(np.interp(z, soc, ocvrel))
        
        return ocv_base + T * ocv_temp
    
    def dOCVfromSOCtemp(self, z, T):
        """
        Compute d(OCV)/d(SOC) at a given SOC (z) and temperature (T).
        Uses central differences on model data.
        Args:
            z: SOC value between 0 and 1
            T: temperature (°C)
        Returns:
            dOCV/dSOC value in V/SOC
        """
        # Input validation
        if not 0 <= z <= 1:
            print(f"Warning: SOC {z:.2%} outside valid range [0,1]. Clamping to valid range.")
            z = np.clip(z, 0, 1)

        soc = np.array(self.model['SOC0'].dropna())
        ocv0 = np.array(self.model['OCV0'].dropna())
        ocvrel = np.array(self.model['OCVrel'].dropna())

        # Ensure all arrays are aligned in length
        min_len = min(len(soc), len(ocv0), len(ocvrel))
        soc, ocv0, ocvrel = soc[:min_len], ocv0[:min_len], ocvrel[:min_len]

        # Calculate derivatives using central differences
        dOCV0_dSOC = np.gradient(ocv0, soc)
        dOCVrel_dSOC = np.gradient(ocvrel, soc)

        # Interpolate derivatives at given SOC
        dOCV0 = float(np.interp(z, soc, dOCV0_dSOC))
        dOCVrel = float(np.interp(z, soc, dOCVrel_dSOC))

        return dOCV0 + T * dOCVrel
        

    
    def iterEKF(self, vk, ik, Tk, deltat):
        
        # model = self.model

        # Load the cell model parameters for the present operating temp
        Q = self.model['QParam'][5]
        G = self.model['GParam'][5]
        M = self.model['MParam'][5]
        M0 = self.model['M0Param'][5]
        RC = self.model['RCParam'][5]
        R = self.model['RParam'][5]
        R0 = self.model['R0Param'][5]
        eta = self.model['etaParam'][5]

        if ik<0: ik=ik*eta # adjust current if charging cell

        # Get data stored in data structure
        SigmaX = self.SigmaX
        SigmaW = self.SigmaW
        SigmaV = self.SigmaV
        irInd = self.irInd
        hkInd = self.hkInd
        zkInd = self.zkInd
        xhat = self.xhat
        nx = xhat.shape[0]
        I = self.priorI
        if abs(ik) > Q/100: self.signIK = np.sign(ik)  # Update sign of current if large enough
        signIK = self.signIK

        # EKF Step 1: State prediction time update
        # First compute Ahat[k-1] and Bhat[k-1]
        Ah = np.exp(-np.abs((I * G * deltat) / (3600 * Q)))
        Bh = - np.abs(G * deltat / (3600 * Q)) * Ah * (1 + np.sign(I) * float(xhat[hkInd, 0]))
        
        Ahat = np.zeros((nx, nx))
        Bhat = np.zeros((nx, 1))

        Ahat[zkInd, zkInd] = 1
        Bhat[zkInd] = - deltat / (3600 * Q)

        Ahat[irInd, irInd] = np.diag([RC])
        Bhat[irInd] = [1 - RC]

        B = np.hstack([Bhat, np.zeros_like(Bhat)])

        Ahat[hkInd, hkInd] = Ah
        Bhat[hkInd, 0] = Bh

        B[hkInd, 1] = Ah - 1

        # Next update xhat
        xhat = Ahat @ xhat + B @ np.array([[I], [np.sign(I)]])

        # EKF Step 2: Error covariance prediction time update
        # sigmaminus(k) = Ahat[k-1] * sigmaplus[k-1] * Ahat[k-1].T 
        #                   + Bhat[k-1] * sigmawtilde * Bhat[k-1].T

        SigmaX = Ahat @ SigmaX @ Ahat.T + Bhat @ np.atleast_2d(SigmaW) @ Bhat.T

        # EKF Step 3: Output estimate

        yhat = self.OCVfromSOCtemp(xhat[zkInd, 0], Tk) + M0 * signIK + M * xhat[hkInd, 0] - R * xhat[irInd, 0] - R0 * ik

        # EKF Step 4: Estimator gain matrix
        Chat = np.zeros((1, nx))
        Chat[0, zkInd] = self.dOCVfromSOCtemp(float(xhat[zkInd, 0]), Tk)
        Chat[0, hkInd] = M
        Chat[0, irInd] = -R
        Dhat = np.array([[1]])
        SigmaY = Chat @ SigmaX @ Chat.T + Dhat * SigmaV
        L = SigmaX @ Chat.T @ np.linalg.inv(SigmaY)

        # EKF Step 5: State estimate measurment update
        r = vk - yhat # residual. Use to check sensor errors...

        r_scalar = float(np.atleast_1d(r).squeeze())
        SigmaY_scalar = float(np.atleast_1d(SigmaY).squeeze())


        if r_scalar**2 > 100 * SigmaY_scalar: 
            L = np.zeros_like(L) # If residual is too large, don't update state
        xhat = xhat + L * r
        xhat[hkInd, 0] = np.minimum(1, np.maximum(-1, xhat[hkInd, 0])) # help maintain robustness
        xhat[zkInd, 0] = np.minimum(1.05, np.maximum(-0.05, xhat[zkInd, 0])) 


        # EKF Step 6: Error covariance measurement update
        SigmaX = SigmaX - L @ SigmaY @ L.T
        #SigmaX bumb code
        if r_scalar**2 > 4 * SigmaY_scalar: # Bad voltage estimate by 2 std, bump SigmaX
            print("Bumping SigmaX")
            SigmaX[zkInd, zkInd] = SigmaX[zkInd, zkInd] * self.SXbump

        #SigmaX = (SigmaX + SigmaX.T) / 2  # Force symmetry
        U, S, V = np.linalg.svd(SigmaX)  # SVD for numerical stability
        HH = V @ S @ V.T
        SigmaX = (SigmaX + SigmaX.T + HH + HH.T) / 4  # help maintain robustness

        # Save data in EKF structure for next time...
        self.priorI = ik
        self.SigmaX = SigmaX
        self.xhat = xhat
        zk = float(xhat[zkInd, 0])
        zkbnd = float(3 * np.sqrt(SigmaX[zkInd, zkInd]))
        
        # Return SOC estimate and bound
        return zk , zkbnd
