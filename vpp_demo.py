'''
vpp 只报量建模    
author: Huahsin Sun    
necessary package: numpy, docplex, matplotlib    
'''
import numpy as np    
from docplex.mp.model import Model    
import matplotlib.pyplot as plt    
from docplex.mp.context import Context    

# Set up the default context for the optimization model




context = Context.make_default_context()    
context.cpus = 4    

# Define PV power generation data for 24 hours    
pv24 = np.array([    
    
])
pv24 = pv24 * 2  # Scale the PV data    

# Define price data for 24 hours    
price = np.array([    
    
])

# Define SOC (State of Charge) parameters for the energy storage system




Ess_origin = 7.5  # Initial SOC 
Ess_max = 15  # Maximum SOC 
pe_max = 3  # Maximum charge/discharge power
Ess_r = 0.995  # Storage efficiency
Yita = 0.95  # Charge/discharge efficiency
Ess_cost_r = 5  # Cost rate for energy storage
time_line = 24  # Time horizon

# Define DG (Distributed Generation) parameters
pg_max = 2  # Maximum power generation
pg_min = 1  # Minimum power generation
pg_cost = [2.4, 35, 3]  # Cost coefficients for power generation
Penalty_r = 10  # Penalty coefficient
rand_ratio = 0.2  # Random ratio for uncertainty
opration = 6  # Operation mode

# Adjust parameters based on the operation mode
if opration == 1:
    pg_max = pg_min = 0
    Ess_max = 0
    # Pure PV
elif opration == 2:
    pv24 = np.zeros(24)
    pe_max = 0
    # Pure DG
elif opration == 3:
    pv24 = np.zeros(24)
    pg_min = pg_max = 0
    # Pure ESS
elif opration == 4:
    pg_max = pg_min = 0
    # PV + ESS
elif opration == 5:
    pe_max = 0
    # DG + PV

# Create the optimization model
mdl = Model(name='vpp投标优化')

# Define variables for the model
pvs = mdl.continuous_var_list(time_line, name='pv', lb=0)
pgs = mdl.continuous_var_list(time_line, name='pg')
socs = mdl.continuous_var_list(time_line, name='soc', lb=0)
pess_in = mdl.continuous_var_list(time_line, name='pess_in', lb=0)
pess_out = mdl.continuous_var_list(time_line, name='pess_out', lb=0)
vpp = mdl.continuous_var_list(time_line, name='vpp')

# Add constraints to the model
for i in range(time_line):
    mdl.add_constraint(pvs[i] <= pv24[i])  # PV power constraint
    mdl.add_constraint(pess_out[i] <= pe_max)  # ESS discharge power constraint
    mdl.add_constraint(pess_in[i] <= pe_max)  # ESS charge power constraint
    mdl.add_constraint(socs[i] <= Ess_max)  # SOC constraint
    mdl.add_constraint(pgs[i] >= pg_min)  # DG minimum power constraint
    mdl.add_constraint(pgs[i] <= pg_max)  # DG maximum power constraint
    mdl.add_constraint(vpp[i] == pvs[i] + pgs[i] + pess_out[i] * Yita - pess_in[i] / Yita)  # VPP power balance

# Initial SOC constraint
mdl.add_constraint(socs[0] == Ess_origin * Ess_r + pess_in[0] * Yita - pess_out[0] / Yita)

# SOC balance constraints for each time step
for k in range(1, time_line):
    mdl.add_constraint(socs[k] == socs[k - 1] * Ess_r + pess_in[k] * Yita - pess_out[k] / Yita)

# Additional constraints for PV + DG + ESS operation
if 1:
    for i in range(time_line):
        mdl.add_constraint(pv24[i] * rand_ratio <= pg_max + pe_max - (pgs[i] + pess_out[i]))
        mdl.add_constraint(pv24[i] * rand_ratio <= pgs[i] - pg_min + pe_max - pess_in[i])

# Define the objective function
objective = mdl.sum(
    -price[k] * vpp[k] +
    pg_cost[0] * pgs[k] * pgs[k] + pg_cost[1] * pgs[k] + pg_cost[2] +
    Ess_cost_r * (pess_in[k] + pess_out[k]) for k in range(24)
)
mdl.minimize(objective)  # Minimize the objective function

# Solve the optimization model
solution = mdl.solve()

# Print the expected revenue
print("期望收益: ", -solution.objective_value)

# Extract the solution values for each variable
pv = [solution.get_value(pvs[i]) for i in range(time_line)]
pg = [solution.get_value(pgs[i]) for i in range(time_line)]
soc = [solution.get_value(socs[i]) for i in range(time_line)]
vpp = [solution.get_value(vpp[i]) for i in range(time_line)]
p_in = [solution.get_value(pess_in[i]) for i in range(time_line)]
p_out = [solution.get_value(pess_out[i]) for i in range(time_line)]

# Create a time axis for plotting
time = list(range(time_line))

# Set up the plot
plt.figure(figsize=(14, 8))

# Plot PV power
plt.plot(time, pv, label='PV', color='tab:blue', linestyle='-', marker='o')

# Plot DG power
plt.plot(time, pg, label='PG', color='tab:orange', linestyle='-', marker='x')

# Plot SOC
plt.plot(time, soc, label='SOC', color='tab:green', linestyle='-', marker='s')

# Plot VPP power
plt.plot(time, vpp, label='VPP', color='tab:red', linestyle='-', marker='d')

# Plot ESS input power
plt.plot(time, p_in, label='P_in', color='tab:purple', linestyle='-', marker='^')

# Plot ESS output power
plt.plot(time, p_out, label='P_out', color='tab:brown', linestyle='-', marker='v')

# Set plot title and labels
plt.title('change')
plt.xlabel('time')
plt.ylabel('P/MV')
plt.grid(True)

# Add legend to the plot
plt.legend()

# Show the plot
plt.show()

# Plot PV power comparison
plt.plot(range(time_line), pv24, label='24')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Power')
plt.title('PV Power Comparison')
plt.pause(2)

# Generate real PV power data with randomness
P_PV_max_real = [pv24[t] * (1 + rand_ratio * (np.random.rand() - 0.5) / 0.5) for t in range(time_line)]

# Create a new optimization model for real-time VPP optimization
mdl_real = Model(name='VPP_real_Optimization')

# Define variables for the real-time model
PV_P_output_real = mdl_real.continuous_var_list(time_line, name='PV_P_output_real')
DG_P_output_real = mdl_real.continuous_var_list(time_line, name='DG_P_output_real')
ESS_P_output_real = mdl_real.continuous_var_list(time_line, name='ESS_P_output_real')
ESS_P_input_real = mdl_real.continuous_var_list(time_line, name='ESS_P_input_real')
SOC_state_real = mdl_real.continuous_var_list(time_line, name='SOC_state_real')
Agent_output_real = mdl_real.continuous_var_list(time_line, name='Agent_output_real')

# Add constraints to the real-time model
for i in range(time_line):
    mdl_real.add_constraint(PV_P_output_real[i] >= 0)
    mdl_real.add_constraint(PV_P_output_real[i] <= P_PV_max_real[i])
    mdl_real.add_constraint(ESS_P_output_real[i] >= 0)
    mdl_real.add_constraint(ESS_P_output_real[i] <= pe_max)
    mdl_real.add_constraint(ESS_P_input_real[i] >= 0)
    mdl_real.add_constraint(ESS_P_input_real[i] <= pe_max)
    mdl_real.add_constraint(SOC_state_real[i] >= 0)
    mdl_real.add_constraint(SOC_state_real[i] <= Ess_max)
    mdl_real.add_constraint(DG_P_output_real[i] >= pg_min)
    mdl_real.add_constraint(DG_P_output_real[i] <= pg_max)
    mdl_real.add_constraint(
        Agent_output_real[i] == PV_P_output_real[i] + DG_P_output_real[i] + ESS_P_output_real[i] * Yita -
        ESS_P_input_real[i] / Yita)

# Initial SOC constraint for the real-time model
mdl_real.add_constraint(SOC_state_real[0] == Ess_origin * Ess_r + ESS_P_input_real[0] * Yita - ESS_P_output_real[0] / Yita)

# SOC balance constraints for each time step in the real-time model
for k in range(1, time_line):
    mdl_real.add_constraint(SOC_state_real[k] == SOC_state_real[k - 1] * Ess_r + ESS_P_input_real[k] * Yita - ESS_P_output_real[k] / Yita)

# Define penalty terms for deviations in the real-time model
Penalty_terms = []
for k in range(time_line):
    deviation = vpp[k] - Agent_output_real[k]
    penalty = mdl_real.continuous_var(name=f'penalty_{k}')
    mdl_real.add_constraint(penalty >= deviation)
    mdl_real.add_constraint(penalty >= -deviation)
    Penalty_terms.append(Penalty_r * price[k] * penalty)
