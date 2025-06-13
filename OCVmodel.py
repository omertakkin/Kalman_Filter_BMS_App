import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read input file
input_file = pd.read_csv('model_input/input_5.csv')
t = np.array(input_file['time'].dropna())     # time array, shape (N,)
i = np.array(input_file['current'].dropna())  # current array, shape (N,)
init_soc = input_file['initsoc'][0]           # initial SOC, scalar value 

# Set simulation length based on current array
dt = t[1] - t[0]                 # assume uniform sampling
sample_freq = 1 / dt
sim_steps = len(i)               # number of simulation steps

# Truncate t to match current
t = t[:sim_steps]

idx_T = 0  # [-25 -15 -5 5 15 25 45] degrees Celsius, index for temperature

# read your cell model once
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


# Initialize arrays to match input data length
z = np.ones(sim_steps)  # initialize z
z[0] = init_soc  # set initial SOC
i_R = np.zeros(sim_steps)  # initialize i_R
h = np.zeros(sim_steps)  # initialize h
v = np.zeros(sim_steps)  # initialize v

s = 0  # Initialize s

# Simulation loop
for time in range(sim_steps - 1):
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

# Debug print statements
print(f"Last SOC (z): {z[-1]:.2f}")

data = pd.DataFrame({
    'time': t,
    'current': i,
    'voltage': v,
    'soc': z,
})
data.to_csv('model_out/sim_data.csv', index=False) # This saves the simulated time, true SOC (z), voltage (v), and current (i) for use by the EKF script.

mins = t / 60  # Convert time to minutes for plotting
# Plotting
fig, ax1 = plt.subplots()

# Plot voltage on left y-axis
ax1.plot(mins, v, label="Output Voltage", color=[0.1529, 0.6824, 1])
ax1.set_xlabel("Time (min)")
ax1.set_ylabel("Voltage (V)", color=[0.1529, 0.6824, 1])
ax1.tick_params(axis='y', labelcolor=[0.1529, 0.6824, 1])
ax1.grid(True)
# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
ax2.plot(mins, i, label="Current", color=[1, 0.3333, 0.2706])
ax2.set_ylabel("Current (A)", color=[1, 0.3333, 0.2706])
ax2.tick_params(axis='y', labelcolor=[1, 0.3333, 0.2706])
# Title and show
plt.title("Output Voltage and Current vs Time")
fig.tight_layout()
plt.show()


plt.plot(mins, z, label="State of charge")
plt.title("State of charge")
plt.xlabel("Time (min)")
plt.ylabel("SOC")
plt.grid(True)
plt.legend()
plt.show()