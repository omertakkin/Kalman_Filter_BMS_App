import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulation variables
dt = 1  # delta t in seconds
sample_freq = 1 / dt
sim_end_time = 6000  # in seconds
curr_T = 5  # Index corresponding -> 25 degrees Celsius

OCV_vs_SOC_0deg = pd.read_csv('model_param/OCV_vs_SOC.csv')
OCV_temp_rel = pd.read_csv('model_param/OCV_temp_var.csv')




def get_OCV0(z):
  OCV0_curr = np.interp(z, SOC0, OCV0)
  return OCV0_curr
def get_OCVrel(z):
  OCVrel_curr = np.interp(z, SOCrel, OCVrel)
  return OCVrel_curr



# Fields pertaining to the OCV versus SOC relationship:
OCV0 = OCV_vs_SOC_0deg.iloc[:,1]    # Vector of OCV versus SOC at 0 degree Celsius
OCVrel = OCV_temp_rel.iloc[:,1]  # Vector of change in OCV versus SOC per degree Celsius [V/C]
SOC = 1     # SOC vector at which OCV0 and OCVrel are stored
SOC0 = OCV_vs_SOC_0deg.iloc[:,0]    # Vector OF SOC versus OCV at 0 degree Celsius
SOCrel = OCV_temp_rel.iloc[:,0]  # Vector of change in SOC versus OCV per degree Celsius [1/C]
OCV = 3.5   # OCV vector at which SOC0 and SOCrel are stored

# Fields pertaining to the dynamic relationship:
temps = np.array([-25, -15, -5, 5, 15, 25, 45])  # Temperatures at which dynamic parameters are stored [C]
Qparam = np.array([25, 25, 25, 25, 25, 25, 25])  # Capacity Q at each temperature [Ah]
etaParam = np.array([1, 1, 1, 1, 1, 1, 1])  # Coulombic efficiency eta at each temperature [unitless]
Gparam = np.array([10, 90, 100, 90, 60, 170, 190])  # Hysteresis "gamma" parameter [unitless]
MParam = np.array([90, 60, 40, 20, 25, 20, 10]) * 0.001  # Hysteresis M parameter [V]
M0Param = np.array([5, 1, 1, 5, 4, 3, 2]) * 0.001  # Hysteresis M0 parameter [V]
R0Param = np.array([10, 8, 5, 4, 3, 2, 1]) * 0.001  # Series resistance parameter R_0 [ohm]
RCParam = np.array([1.5, 1.3, 1.2, 2.2, 3.4, 3.5, 3.4])  # The R-C time constant parameter R_j C_j [s]
RParam = np.array([5, 2, 1, 1, 0.8, 0.5, 0.3]) * 0.001  # Resistance R_j of R-C parameter [ohm]

# Inputs
i = np.ones(int(sim_end_time / dt) + 1)  # initialize i
i[:300] = 30
i[300:900] = -5   # 25A for 600 seconds
i[900:] = 0

# States
z = np.ones(int(sim_end_time / dt + 1))  # initialize z
i_R = np.zeros(int(sim_end_time / dt + 1))  # initialize i_R
h = np.zeros(int(sim_end_time / dt + 1))  # initialize h
# Output
v = np.zeros(int(sim_end_time / dt + 1))  # initialize v

s = 0  # Initialize s

# Simulation loop
for time in range(int(sim_end_time)):
    # calculating next state 
    A_RC = np.exp(-dt / RCParam[curr_T])
    B_RC = 1 - A_RC
    A_H = np.exp(-np.abs((etaParam[curr_T] * i[time] * Gparam[curr_T] * dt) / (Qparam[curr_T]*3600)))
    z[time + 1] = z[time] + (-etaParam[curr_T] * dt / (Qparam[curr_T]*3600)) * i[time]
    i_R[time + 1] = A_RC * i_R[time] + B_RC * i[time]
    h[time + 1] = A_H * h[time] + (A_H - 1) * np.sign(i[time])

    # calculating current output
    if abs(i[time]) > 0:
        s = np.sign(i[time])
    OCV = get_OCV0(z[time]) + get_OCVrel(z[time])*0.001*temps[curr_T]
    v[time + 1] = OCV + M0Param[curr_T]*s + MParam[curr_T]*h[time] - RParam[curr_T]*i_R[time] - R0Param[curr_T]*i[time]

time_array = np.arange(0, sim_end_time+dt ,dt)


data = pd.DataFrame({
    'time': time_array,
    'SOC_true': z,
    'voltage': v,
    'current': i
})
data.to_csv('model_out/sim_data.csv', index=False) # This saves the simulated time, true SOC (z), voltage (v), and current (i) for use by the EKF script.


# Plotting
plt.plot(time_array, v, label="Output Voltage")
plt.title("Output Voltage (V)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
#plt.xlim(250,2000)
#plt.ylim(3.2,3.6)
plt.legend()
plt.show()


plt.plot(time_array, z, label="State of charge")
plt.title("State of charge")
plt.xlabel("Time (s)")
plt.ylabel("SOC")
plt.grid(True)
plt.legend()
plt.show()