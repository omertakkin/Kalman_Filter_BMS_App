import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulation variables
dt = 1  # delta t in seconds
sample_freq = 1 / dt
sim_end_time = 100  # in seconds
idx_T = 5  # [-25 -15 -5 5 15 25 45] degrees Celsius, index for temperature

cell_model = pd.read_csv('model_param/cell_model.csv')

# Fields pertaining to the OCV versus SOC relationship:
OCV   = cell_model['OCV'].dropna()        # OCV vector at which SOC0 and SOCrel are stored
OCV0  = cell_model['OCV0'].dropna()       # Vector of OCV versus SOC at 0 degree Celsius
OCVrel= cell_model['OCVrel'].dropna()     # Vector of change in OCV versus SOC per degree Celsius [V/C]
SOC   = cell_model['SOC'].dropna()        # SOC vector at which OCV0 and OCVrel are stored
SOC0  = cell_model['SOC0'].dropna()       # Vector OF SOC versus OCV at 0 degree Celsius
SOCrel= cell_model['SOCrel'].dropna()     # Vector of change in SOC versus OCV per degree Celsius [1/C]

# Fields pertaining to the dynamic relationship:
temps   = cell_model['temps'].dropna()    # Temperatures at which dynamic parameters are stored [C]
QParam  = cell_model['QParam'].dropna()   # Capacity Q at each temperature [Ah]
etaParam= cell_model['etaParam'].dropna() # Coulombic efficiency eta at each temperature [unitless]
GParam  = cell_model['GParam'].dropna()   # Hysteresis "gamma" parameter [unitless]
MParam  = cell_model['MParam'].dropna()   # Hysteresis M parameter [V]
M0Param = cell_model['M0Param'].dropna()  # Hysteresis M0 parameter [V]
R0Param = cell_model['R0Param'].dropna()  # Series resistance parameter R_0 [ohm]
RCParam = cell_model['RCParam'].dropna()  # The R-C time constant parameter R_j C_j [s]
RParam  = cell_model['RParam'].dropna()   # Resistance R_j of R-C parameter [ohm]


def get_OCV0(z):
  OCV0_curr = np.interp(z, SOC0, OCV0)
  return OCV0_curr
def get_OCVrel(z):
  OCVrel_curr = np.interp(z, SOCrel, OCVrel)
  return OCVrel_curr

# Inputs
i = np.ones(int(sim_end_time / dt) + 1)  # initialize i
i[:25] = 300
i[25:75] = -50   # 25A for 600 seconds
i[57:] = 0

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
    A_RC = np.exp(-dt / RCParam[idx_T])
    B_RC = 1 - A_RC
    A_H = np.exp(-np.abs((etaParam[idx_T] * i[time] * GParam[idx_T] * dt) / (QParam[idx_T]*3600)))
    z[time + 1] = z[time] + (-etaParam[idx_T] * dt / (QParam[idx_T]*3600)) * i[time]
    i_R[time + 1] = A_RC * i_R[time] + B_RC * i[time]
    h[time + 1] = A_H * h[time] + (A_H - 1) * np.sign(i[time])

    # calculating current output
    if abs(i[time]) > 0:
        s = np.sign(i[time])
    OCV = get_OCV0(z[time]) + get_OCVrel(z[time])*0.001*temps[idx_T]
    v[time + 1] = OCV + M0Param[idx_T]*s + MParam[idx_T]*h[time] - RParam[idx_T]*i_R[time] - R0Param[idx_T]*i[time]

t = np.arange(0, sim_end_time+dt ,dt)


data = pd.DataFrame({
    'time': t,
    'current': i,
    'voltage': v,
    'soc': z,
})
data.to_csv('model_out/sim_data.csv', index=False) # This saves the simulated time, true SOC (z), voltage (v), and current (i) for use by the EKF script.


# Plotting
plt.plot(t, v, label="Output Voltage")
plt.title("Output Voltage (V)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
#plt.xlim(250,2000)
#plt.ylim(3.2,3.6)
plt.legend()
plt.show()


plt.plot(t, z, label="State of charge")
plt.title("State of charge")
plt.xlabel("Time (s)")
plt.ylabel("SOC")
plt.grid(True)
plt.legend()
plt.show()