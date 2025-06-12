from EKF import EKF 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cell_model = pd.read_csv('model_param/cell_model.csv')
Cell_DYN_P5 = pd.read_csv('model_out/sim_data.csv')
T = 25  # degrees Celsius

time = Cell_DYN_P5['time'].values[1:]       # Time       
time = time - time[0]                       # Normalize time to start from 0
deltat = time[1] - time[0]                  # Sample interval
current = Cell_DYN_P5['current'].values[1:] # Current , discharge> 0;charge <0
voltage = Cell_DYN_P5['voltage'].values[1:] # Voltage
soc = Cell_DYN_P5['soc'].values[1:]         # True SOC

# Reserve storage for computed results, for plotting
sochat = np.zeros_like(soc)
socbound = np.zeros_like(soc)

# Covariance values
SigmaX0 = np.diag([1e-6, 1e-8, 2e-4])   # uncertainty in initial state
SigmaW = 2e-1   # uncertainty in current sensor, state equation
SigmaV = 2e-1   # uncertainty in voltage sensor, output equation

# Create ekfData structure and initialize variables using first voltage measurement and first temperature measurement
EKF_model = EKF( voltage[0] , T , SigmaX0, SigmaV, SigmaW , cell_model )

# Now enter a loop for remainder of time, where we update the EKF once per sample interval

for k in range(voltage.shape[0] - 1):

    vk = voltage[k] # 'measure' voltage
    ik = current[k] # 'measure' current
    Tk = T          # 'measure' temperature

    # Update SOC (and model state)
    sochat[k],socbound[k] = EKF_model.iterEKF(vk,ik,Tk,deltat)
    print(f"Iteration {k+1}/{voltage.shape[0]-1}: SOC = {sochat[k]:.4f}, Bound = {socbound[k]:.4f}")

# Plot results

# Plot 1: SOC estimation
plt.figure(1)
plt.clf()
plt.plot(time / 60, 100 * sochat, label='Estimate')
plt.plot(time / 60, 100 * soc, label='Truth')

# Plot bounds
plt.plot(np.concatenate((time / 60, [np.nan], time / 60)),
         np.concatenate((100 * (sochat + socbound), [np.nan], 100 * (sochat - socbound))),
         label='Bounds')

plt.title('SOC estimation using EKF')
plt.xlabel('Time (min)')
plt.ylabel('SOC (%)')
plt.legend()
plt.grid(True)

# Print RMS error
rms_error = np.sqrt(np.mean((100 * (soc - sochat))**2))
print(f'RMS SOC estimation error = {rms_error:.4g}%')

# Plot 2: SOC estimation error
plt.figure(2)
plt.clf()
plt.plot(time / 60, 100 * (soc - sochat), label='Estimation error')

# Plot error bounds
plt.plot(np.concatenate((time / 60, [np.nan], time / 60)),
         np.concatenate((100 * socbound * np.ones_like(sochat), [np.nan], -100 * socbound * np.ones_like(sochat))),
         label='Bounds')

plt.title('SOC estimation errors using EKF')
plt.xlabel('Time (min)')
plt.ylabel('SOC error (%)')
plt.ylim([-4, 4])
plt.legend()
plt.grid(True)

# Compute percentage of time error outside bounds
ind = np.where(np.abs(soc - sochat) > socbound)[0]
percent_outside = len(ind) / len(soc) * 100
print(f'Percent of time error outside bounds = {percent_outside:.4g}%')

plt.show()