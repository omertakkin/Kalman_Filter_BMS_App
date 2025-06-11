import numpy as np
import matplotlib.pyplot as plt

# Initialize simulation variables
SigmaW = np.array([[1.0]])     # Process noise covariance (Q)
SigmaV = np.array([[2.0]])     # Sensor noise covariance (R)
maxIter = 100

xtrue = np.array([[2.0 + np.random.randn()]])   # True initial state
xhat = np.array([[2.0]])                        # Estimated initial state
SigmaX = np.array([[1.0]])                      # Initial error covariance
u = np.array([[0.0]])                           # Input (not used here)

# Reserve storage for plotting
xstore = np.zeros((maxIter + 1, len(xtrue))) 
xstore[0, :] = xtrue
xhatstore = np.zeros((maxIter, len(xhat)))
SigmaXstore = np.zeros((maxIter, len(xhat)**2))
errorstore = np.zeros((maxIter, len(xtrue)))

for k in range(maxIter):

    # EKF Step 1: Predict state using f(x) = sqrt(5 + x)
    Ahat = np.array([[0.5 / np.sqrt(5 + xhat[0, 0]) ]])  
    xhat = np.array([[np.sqrt(5 + xhat[0, 0])]])   

    # EKF Step 2: Predict covariance
    SigmaX_pred = Ahat @ SigmaX @ Ahat.T + SigmaW

    # Simulate true system and measurement
    w = np.linalg.cholesky(SigmaW) @ np.random.randn(1, 1)
    v = np.linalg.cholesky(SigmaV) @ np.random.randn(1, 1)
    xtrue = np.array([[np.sqrt(5 + xtrue[0, 0])]]) + w
    y = xtrue[0, 0]**3 + v

    # EKF Step 3: Predict measurement
    Chat = np.array([[3 * xhat[0, 0]**2]])         # dh/dx at predicted x
    yhat = np.array([[xhat[0, 0]**3]])             # Predicted measurement

    # EKF Step 4: Kalman gain
    SigmaY = Chat @ SigmaX_pred @ Chat.T + SigmaV
    L = SigmaX_pred @ Chat.T @ np.linalg.inv(SigmaY)

    # EKF Step 5: Update state estimate
    xhat = xhat + L @ (y - yhat)
    xhat[0, 0] = np.maximum(-5, xhat[0, 0])

    # EKF Step 6: Update covariance (Joseph form for stability)
    I = np.eye(1)
    SigmaX = (I - L @ Chat) @ SigmaX_pred @ (I - L @ Chat).T + L @ SigmaV @ L.T

    # Store for plotting
    xstore[k + 1, :] = xtrue
    xhatstore[k, :] = xhat
    SigmaXstore[k, :] = SigmaX.flatten()
    errorstore[k, :] = np.abs(xtrue - xhat)

# Plotting
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(xstore, label='True State', color='blue')
plt.plot(xhatstore, label='Estimated State', color='orange')
plt.fill_between(range(maxIter),
                 xhatstore[:, 0] - np.sqrt(SigmaXstore[:, 0]),
                 xhatstore[:, 0] + np.sqrt(SigmaXstore[:, 0]),
                 color='lightgray', alpha=0.5, label='Uncertainty')
plt.title('Extended Kalman Filter State Estimation')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(errorstore, label='Estimation Error', color='red')
plt.title('Estimation Error Over Time')
plt.xlabel('Time Step')
plt.ylabel('Absolute Error')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()