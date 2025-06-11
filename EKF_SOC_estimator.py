import numpy as np
import pandas as pd

# Load simulated data from OCVmodel
data = pd.read_csv('model_out/sim_data.csv')
time = data['time'].values[1:]              # skip t=0 if needed
v_meas = data['voltage'].values[1:]
I = data['current'].values[1:]
SOC_true = data['SOC_true'].values[1:]

# Reload OCV vs SOC table for measurement model
OCV_vs_SOC = pd.read_csv('model_param/OCV_vs_SOC.csv')
SOC_table = OCV_vs_SOC.iloc[:,0]
OCV_table = OCV_vs_SOC.iloc[:,1]

# Battery parameters
Q = 25.0            # battery capacity [Ah]
eta = 1.0           # coulombic efficiency
G = 170.0           # hysteresis parameter (from Gparam at 25°C index)
M = 0.020           # hysteresis M (from MParam at 25°C index)
M0 = 0.003          # hysteresis M0 (from M0Param at 25°C index)
R = 0.0005         # series resistance R (from RParam at 25°C index)
R0 = 0.002         # ohmic resistance R0 (from R0Param at 25°C index)
RC = 3.5           # R-C time constant (from RCParam at 25°C index)

"""
SigmaW = np.diag([1e-7, 1e-8, 1e-8]) # Process noise covariance
SigmaV = 1e-4                        # Measurement noise variance (voltage)
maxIter = len(time) - 1  # Number of iterations based on data length

xtrue = np.array([0.95, 0.0, 0.0])  # True initial state: [SOC, i_R, h]
xhat = np.array([0.95, 0.0, 0.0])  # Initial estimate: e.g. 95% SOC, no internal states
SigmaX = np.diag([1e-4, 1e-6, 1e-6])  # Initial error covariance
u = np.zeros((1, 1))  # Unkown initial driving input (assume zero)

# Reserve storage for variables
xstore = np.zeros((maxIter + 1, len(xtrue)))
xstore[0, :] = xtrue
xhatstore = np.zeros((maxIter, len(xhat)))
SigmaXstore = np.zeros((maxIter, len(xhat)**2))
errorstore = np.zeros((maxIter, len(xtrue)))

for k in range(maxIter):

    # Step 1: State prediction time update
    Ahat = 

"""






# Number of data points
N = len(time)

# Initialize state and covariance
# State vector x = [SOC, i_R, h]
x_est = np.zeros((3, N))
x_est[:,0] = [0.95, 0.0, 0.0]  # initial guess: e.g. 95% SOC, no internal states
P = np.zeros((3,3,N))
P[:,:,0] = np.diag([1e-4, 1e-6, 1e-6])  # small initial uncertainties

# Process and measurement noise (tunable)
Q_k = np.diag([1e-7, 1e-8, 1e-8])   # process noise covariance
R_k = 1e-4                          # measurement noise variance (voltage)

for k in range(1, N):
    # === Predict step ===
    soc, iR, h = x_est[:,k-1]
    Ik = I[k-1]
    # Dynamics coefficients
    A_RC = np.exp(-1.0/RC)
    B_RC = 1.0 - A_RC
    if abs(Ik) > 0:
        A_H = np.exp(-abs((eta*Ik*G)/(Q*3600.0)))
        s = np.sign(Ik)
    else:
        A_H = 1.0; s = 0.0
    # State prediction (coulomb counting + RC + hysteresis)
    soc_pred = soc - (eta*Ik)/(Q*3600.0)
    iR_pred = A_RC*iR + B_RC*Ik
    h_pred = A_H*h + (A_H - 1.0)*s
    x_pred = np.array([soc_pred, iR_pred, h_pred])
    # Jacobian F = df/dx (linearized state transition)
    F = np.array([[1.0, 0.0,  0.0],
                  [0.0, A_RC, 0.0],
                  [0.0, 0.0,  A_H]])
    P_pred = F @ P[:,:,k-1] @ F.T + Q_k

    # === Update step with voltage measurement ===
    # Compute predicted voltage from state prediction
    # (Linear interpolation for OCV)
    OCV_pred = np.interp(x_pred[0], SOC_table, OCV_table)
    v_pred = OCV_pred + M0*s + M*x_pred[2] - R*x_pred[1] - R0*Ik

    # Jacobian H = dh/dx of voltage wrt state
    # We numerically compute dOCV/dSOC
    eps = 1e-6
    dOCV_dz = (np.interp(x_pred[0]+eps, SOC_table, OCV_table) -
               np.interp(x_pred[0]-eps, SOC_table, OCV_table)) / (2*eps)
    H = np.array([[dOCV_dz, -R, M]])

    # Kalman gain
    S = H @ P_pred @ H.T + R_k
    K = P_pred @ H.T @ np.linalg.inv(S)
    # Measurement residual
    y = v_meas[k]
    y_pred = v_pred
    x_new = x_pred + (K.flatten() * (y - y_pred))
    P_new = (np.eye(3) - K @ H) @ P_pred

    # Save updated state
    x_est[:,k] = x_new
    P[:,:,k] = P_new

# Extract SOC estimates and compute error
soc_est = x_est[0,:]
soc_error = soc_est - SOC_true

# (Plotting code below would generate the requested figures)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(time, SOC_true, label="True SOC")
plt.plot(time, soc_est, '--', label="Estimated SOC")
plt.xlabel("Time (s)"); plt.ylabel("State of Charge (SOC)")
plt.title("True vs. Estimated SOC"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('estimation_out/soc_estimation_plot.png')

plt.figure(figsize=(6,4))
plt.plot(time, soc_error, label="SOC Estimation Error")
plt.xlabel("Time (s)"); plt.ylabel("Error (SOC units)")
plt.title("SOC Estimation Error (Estimated – True)"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('estimation_out/soc_error_plot.png')
