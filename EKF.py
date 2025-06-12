import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class EKF:
    def __init__(self, v0, idx_T0, SigmaX0, SigmaV, SigmaW, model):
        
        # Store model
        self.model = model

        # Extract only valid pairs for OCV0/SOC0 and OCVrel/SOCrel
        mask0 = (~self.model['SOC0'].isna()) & (~self.model['OCV0'].isna())
        self.SOC0 = np.array(self.model.loc[mask0, 'SOC0'])
        self.OCV0 = np.array(self.model.loc[mask0, 'OCV0'])

        maskrel = (~self.model['SOCrel'].isna()) & (~self.model['OCVrel'].isna())
        self.SOCrel = np.array(self.model.loc[maskrel, 'SOCrel'])
        self.OCVrel = np.array(self.model.loc[maskrel, 'OCVrel'])

        self.temps = np.array(self.model['temps'].dropna())
        self.idx_T0 = idx_T0
        T0 = self.temps[idx_T0]  
        
        # Initialize state description
        ir0 = 0.0  # Initial diffusion current
        hk0 = 0.0  # Initial hysteresis voltage
        SOC0 = self.SOCfromOCVtemp(v0, T0)
        
        """
        ### Debugging
        print(f"\nInitial voltage: {v0}, Initial SOC from OCV: {SOC0}")
        soc_range = np.linspace(0, 1, 100)
        ocv_curve = [self.OCVfromSOCtemp(z, T0) for z in soc_range]
        plt.plot(soc_range, ocv_curve)
        plt.xlabel('SOC')
        plt.ylabel('OCV')
        plt.title('OCV vs SOC')
        plt.show()
        """

        # State variable indices
        self.irInd = 0
        self.hkInd = 1
        self.zkInd = 2

        # Initial state (column matrix)
        self.xhat = np.array([[ir0] , [hk0] , [SOC0]])

        # Covariances - ensure they are positive definite
        self.SigmaW = np.abs(SigmaW)  # Ensure positive
        self.SigmaV = np.abs(SigmaV)  # Ensure positive
        self.SigmaX = np.diag(np.diag(SigmaX0))  # Use only diagonal elements initially
        self.SXbump = 5

        # Previous current value
        self.priorI = 0.0
        self.signIK = 0.0
        
        # Add minimum covariance values to prevent numerical issues
        self.min_cov = 1e-6

    def OCVfromSOCtemp(self, z, T):
        """
        Get temperature-compensated OCV from SOC (z) and T (°C)
        """
        # Interpolate OCV0 at given SOC
        ocv0 = np.interp(z, self.SOC0, self.OCV0)
        
        # Interpolate OCVrel at given SOC
        ocvrel = np.interp(z, self.SOCrel, self.OCVrel)
        
        # If OCV0 is at 0°C, use T directly. If at 25°C, use (T-25)
        ocv = ocv0 + ocvrel * 0.001 * T
        
        return ocv

    def SOCfromOCVtemp(self, v, T):
        """
        Estimate SOC from OCV and temperature using interpolation
        """
        # Create a function to find SOC that gives the target OCV
        def find_soc(soc_guess):
            return self.OCVfromSOCtemp(soc_guess, T) - v
            
        # Try different initial guesses
        initial_guesses = [0.2, 0.5, 0.8]
        best_soc = None
        min_error = float('inf')
        
        for guess in initial_guesses:
            try:
                soc = fsolve(find_soc, guess, full_output=True)
                if soc[1]['fvec'][0] < min_error:
                    min_error = soc[1]['fvec'][0]
                    best_soc = soc[0][0]
            except:
                continue
                
        if best_soc is None:
            # Fallback to linear interpolation if root finding fails
            # Find the closest OCV values in our lookup table
            ocv0_values = self.OCV0
            soc0_values = self.SOC0
            best_soc = np.interp(v, ocv0_values, soc0_values)
        
        # Ensure SOC is within valid range [0, 1]
        soc = np.clip(best_soc, 0, 1)
        
        return soc

    def dOCVfromSOCtemp(self, z, T):
        """
        Compute dOCV/dSOC at a given SOC z and temperature T
        """
        # Small perturbation for numerical derivative
        delta = 1e-6
        
        # Calculate OCV at slightly higher and lower SOC
        ocv_plus = self.OCVfromSOCtemp(z + delta, T)
        ocv_minus = self.OCVfromSOCtemp(z - delta, T)
        
        # Compute numerical derivative
        dOCV = (ocv_plus - ocv_minus) / (2 * delta)
        
        return dOCV

    
    def iterEKF(self, vk, ik, Tk, deltat):
        
        # Load the cell model parameters for the present operating temp
        Q = self.model['QParam'][self.idx_T0]
        G = self.model['GParam'][self.idx_T0]
        M = self.model['MParam'][self.idx_T0]
        M0 = self.model['M0Param'][self.idx_T0]
        RC = self.model['RCParam'][self.idx_T0]
        R = self.model['RParam'][self.idx_T0]
        R0 = self.model['R0Param'][self.idx_T0]
        eta = self.model['etaParam'][self.idx_T0]

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
        SigmaX = Ahat @ SigmaX @ Ahat.T + Bhat @ np.atleast_2d(SigmaW) @ Bhat.T
        
        # EKF Step 3: Output estimate
        yhat = self.OCVfromSOCtemp(xhat[zkInd, 0], Tk) + M0 * signIK + M * xhat[hkInd, 0] - R * xhat[irInd, 0] - R0 * ik

        # EKF Step 4: Estimator gain matrix
        Chat = np.zeros((1, nx))
        Chat[0, zkInd] = self.dOCVfromSOCtemp(float(xhat[zkInd, 0]), Tk)
        Chat[0, hkInd] = M
        Chat[0, irInd] = -R
        Dhat = np.array([[1.0]])
        
        # Ensure numerical stability in covariance calculations
        SigmaY = Chat @ SigmaX @ Chat.T + Dhat * SigmaV
        
        L = SigmaX @ Chat.T @ np.linalg.inv(SigmaY)

        """
        try:
            L = SigmaX @ Chat.T @ np.linalg.inv(SigmaY)
        except np.linalg.LinAlgError:
            # If matrix inversion fails, use a more stable approach
            L = SigmaX @ Chat.T / SigmaY
        """

        # EKF Step 5: State estimate measurement update
        r = vk - yhat  # residual
        
        # Adaptive measurement update
        r_scalar = float(np.atleast_1d(r).squeeze())
        SigmaY_scalar = float(np.atleast_1d(SigmaY).squeeze())

        if r_scalar**2 > 100 * SigmaY_scalar:
            L = np.zeros_like(L)  # Zero gain if residual is too large
        
        xhat = xhat + L * r
        xhat[hkInd, 0] = np.minimum(1, np.maximum(-1, xhat[hkInd, 0]))
        xhat[zkInd, 0] = np.minimum(1.05, np.maximum(-0.05, xhat[zkInd, 0]))
        
        # EKF Step 6: Error covariance measurement update
        SigmaX = SigmaX - L @ SigmaY @ L.T

        # More gradual adaptation based on residual
        if r_scalar**2 > 4 * SigmaY_scalar:
            print("Bumping SigmaX")
            SigmaX[zkInd, zkInd] = SigmaX[zkInd, zkInd] * self.SXbump
        
        # Force symmetry and positive definiteness
        SigmaX = (SigmaX + SigmaX.T) / 2
        eigvals = np.linalg.eigvals(SigmaX)
        if np.any(eigvals < 0):
            SigmaX = SigmaX + np.eye(nx) * self.min_cov
        
        """
        U, S, V = np.linalg.svd(SigmaX)
        HH = V @ S @ V.T
        SigmaX = (SigmaX + SigmaX.T + HH + HH.T) / 4
        np.fill_diagonal(SigmaX, np.maximum(np.diag(SigmaX), self.min_cov))
        """

        # Save data in EKF structure for next time
        self.priorI = ik
        self.SigmaX = SigmaX
        self.xhat = xhat
        zk = float(xhat[zkInd, 0])
        zkbnd = float(3 * np.sqrt(max(SigmaX[zkInd, zkInd], self.min_cov)))
        
        return zk, zkbnd
