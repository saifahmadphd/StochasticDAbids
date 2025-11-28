"""
Description: COde for battery optimal dispatch considering buy and seel to the grid, 
pv and local load.
Code running wiht flexibility, no up down simultaneous flex request.

Testing for flexibility grouping

Working Pch_dch based on soc
"""
# imports
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt


##Body==========================
# rng = np.random.default_rng(seed=455)
rng = np.random.default_rng() # without seed to preserve randomness in each scenario, need to define seed for all scenarios
no_of_days = 1
no_of_scenarios = 500
scenarios = range(no_of_scenarios)
probability = 1 / no_of_scenarios

T = 24 * no_of_days # time horizon for the simulation

time_steps = range(T) 

delta_t = 1 # step size in hours

# Battery specs
E_max_fixed = 98.0  # MWh (Max Energy)
E_min_fixed = 2.0   # MWh (Min Energy based on DoD)
E_initial   = 30.0   # MWh (starting SOC)
P_battery_max  = 40.0   # MW  (charge / discharge cap)
P_grid_max  = 120.0  # grid draw/feed-in power limit MW
efficiency_ch = 0.95
efficiency_dch = 0.96

# Battery EMS behaviour embeddings
alpha_p_set = np.array([0, 0.5, 1.0, 1.0, 0.5, 0.0])
soc_set = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
N_SOC = len(soc_set)

# Market bid limit

E_bid_max_DA = 100 # MWh max bid limit in the DA market

E_bid_max_ID = 60 # MWh max bid limit in the DA market

# system input time series
p_pv_baseline = np.array([
    0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 60, 70,
    60, 50, 40, 30, 20, 10, 0, 0, 0, 0, 0, 0
    ]*no_of_days) *0.5    # pv baseline

p_pv_real = np.array([
    s + (rng.random() - 0.5) * 30 if s >=4 else s 
    for s in p_pv_baseline
    ])

p_pv_scenarios = { scen :
    np.array([s + (rng.uniform(low=-10, high =10))  if s >=10 else s 
    for s in p_pv_baseline])
    for scen in scenarios
    }

#Grid Export restricted times
p_grid_export_cap = np.array([100 if 10<=t%24<=14 else P_grid_max for t in time_steps])


#Flexibility (load UP / down) group availibility

p_flex_up_max = 0  # Max. up flexibility in the hours avl.

p_flex_down_max = 4 # maximum down flex. in the hrs avl.

p_flex_cap_up = np.array([0 if t%24<=9 or t%24>=18 else p_flex_up_max for t in time_steps]) # Increase load in morning during peak PV and export block to grid

p_flex_cap_down = np.array([0 if  8<=t%24<=16 else p_flex_down_max for t in time_steps]) # reduce load during evening no PV

flex_time_max = 1 # in hours

flex_comp_delay = 3 # in hours

flex_share_of_load_max = 0.2 # maximum amount of flex to be requested of total load
#Load
p_load_baseline = np.array([ rng.random() * 20 + 30 if 16 <= t%24 <= 20 else rng.random() * 20 + 20 for t in time_steps ]) #adding higher loads towards the evening

p_load_real = np.array([ 
    s + (rng.random() - 0.5) * 20
    for s in p_load_baseline
    ])
#Energy buy and sell tariff
pi_buy_DA = np.array([8 if 11 <= t%24 < 14 else 15 for t in time_steps]) # electricity buy price forecast in DA

pi_sell_DA = np.array([
    6 if 11 <= t%24 < 14 else 
    12 if 16 <=t%24 < 20 else
    20 if 20 <=t%24 < 22 else
    10    
    for t in time_steps]) # electricity selling price forecast in DA

pi_sell_DA = np.array([ s * (rng.random() - 0.5) * 8 for s in pi_sell_DA ])

pi_buy_ID = np.array([17 if 11 <= t%24 < 14 else 15 for t in time_steps]) # electricity buy price forecast in ID

pi_buy_ID = np.array([ s * (rng.random() - 0.5) * 10 for s in pi_buy_ID ])

pi_sell_ID = np.array([3 if 11 <= t%24 < 14 else 12 for t in time_steps]) # electricity selling price forecast in ID

# Imbalance penalty
penalty_positive_imbalance = np.array([30 if 11 <= t%24 < 14 else 100 for t in time_steps])

penalty_negative_imbalance = np.array([30 if 11 <= t%24 < 14 else 100 for t in time_steps])

# Cost for battery degradation

pi_battery_degradation = 2 # battery degradation cost in Euros/MWh, indicative

#For the FCR DA bids
H_block = 4 #4 hr block in the FCR cooperation

h_activation = 1/3 # 20 minutes actiation energy to be available

pi_fcr = np.array([15 if (11 <= t%24 < 14) or (18 <= t%24 <=23) else 2 for t in time_steps]) 

# 4-hour cooperation blocks
num_blocks = (T + H_block - 1) // H_block   



#Sudden congestion in the grid between 10 and 14 so no export in this time for ID====================================

p_grid_export_cap = np.array([0 if 11<=t%24<=13 else P_grid_max for t in time_steps])

#Gurobi model ID optimization====================================

model = gp.Model("battery_scheduling_ID")

model.setParam("OutputFlag",0) # set to 1 to display the solver o/p

#Scenario dependent decision variables

R_blk = model.addVars(num_blocks, lb=0, ub=P_battery_max, vtype=GRB.CONTINUOUS, name="R_FCR_blk")

E = model.addVars(no_of_scenarios, T+1, lb=E_min_fixed, ub=E_max_fixed, vtype=GRB.CONTINUOUS, name="E") # Energy state

P_ch = model.addVars(no_of_scenarios, T, lb=0, ub=P_battery_max, vtype=GRB.CONTINUOUS, name="P_ch") # Charging power

P_dch = model.addVars(no_of_scenarios, T, lb=0, ub=P_battery_max, vtype=GRB.CONTINUOUS, name="P_dch") # Discharging Power

P_grid_import = model.addVars(no_of_scenarios, T, lb=0, ub=P_grid_max, name="P_grid_import") # Power bought from grid

P_grid_export = model.addVars(no_of_scenarios, T, lb=0, ub=P_grid_max, name="P_grid_export") # Power sold to the grid

P_flex_up = model.addVars(no_of_scenarios, T, lb=0, ub=10, name="P_flexUP") # Power flexibility in Up dir

P_flex_down = model.addVars(no_of_scenarios, T, lb=0, ub=10, name="P_flexDOWN") # Power flexibility in down dir

P_comp_down = model.addVars(no_of_scenarios, T, lb=0, ub=10, name="P_compDOWN")

P_comp_up = model.addVars(no_of_scenarios, T, lb=0, ub=10, name="P_compUP")

E_buy_ID = model.addVars(no_of_scenarios, T, lb=0 , ub=E_bid_max_ID, name="E_buy_ID")

E_sell_ID = model.addVars(no_of_scenarios, T, lb=0 , ub=E_bid_max_ID, name="E_sell_ID")

E_imbalance_positive = model.addVars(no_of_scenarios, T, lb=0, ub=E_bid_max_DA+E_bid_max_ID, name="Imbalance_positive")

E_imbalance_negative = model.addVars(no_of_scenarios, T, lb=0, ub=E_bid_max_DA+E_bid_max_ID, name="Imbalance_negative")

P_fcr_capacity = model.addVars(T, lb=0, ub=25, vtype=GRB.INTEGER, name = "P_fcr_capacity")

lambda_p_soc = model.addVars(no_of_scenarios, T, N_SOC, lb=0, ub=1, name="lambda_p_soc")

alpha_p_factor = model.addVars(no_of_scenarios, T, lb=0, ub=1, name="alpha_p_factor" )

z_bat = model.addVars(no_of_scenarios, T, vtype=GRB.BINARY, name="selection_var_bat") # for selecting ch and dch states

z_grid = model.addVars(no_of_scenarios, T, vtype=GRB.BINARY, name="selection_var_grid") # for buy and sell state selection

z_flex_up = model.addVars(no_of_scenarios, T, vtype=GRB.BINARY, name="up Selection")

z_flex_down = model.addVars(no_of_scenarios, T, vtype=GRB.BINARY, name="down Selection")

z_DA_buy = model.addVars(T, vtype=GRB.BINARY, name="DA buy Selection")

z_DA_sell = model.addVars(T, vtype=GRB.BINARY, name="DA sell Selection")

z_imbalance_positive = model.addVars(no_of_scenarios, T, vtype=GRB.BINARY, name="Positive imbalance Selection")

z_imbalance_negative = model.addVars(no_of_scenarios, T, vtype=GRB.BINARY, name="Negative imbalance Selection")

# non scenario dependent decision variables

E_buy_DA = model.addVars(T, lb=0 , ub=E_bid_max_DA, name="E_buy_DA")

E_sell_DA = model.addVars(T, lb=0 , ub=E_bid_max_DA, name="E_sell_DA")
# Constraints=======================================

for t in time_steps:
    
    b = t // H_block
    model.addConstr(
        P_fcr_capacity[t] == R_blk[b], 
        name=f"FCR_block_link[{t}]"
        )
    model.addConstr(
        z_DA_buy[t] +z_DA_sell[t] <= 1,
        name = "Buy Sell selection"
        )

for scen in scenarios: 
    #Energy Initialization
    model.addConstr(    
        E[scen, 0] == E_initial, 
        name="Initial_Energy"
        )
    
    for t in time_steps:
        
        
        model.addConstr(
            P_ch[scen, t]  <= P_battery_max * z_bat[scen, t],
            name=f"selection1-{t}"
            )
        
        model.addConstr(
            P_dch[scen, t]  <= P_battery_max * (1-z_bat[scen, t]),
            name=f"selection2-{t}"
            )
        
        model.addConstr(
            P_ch[scen, t] + P_fcr_capacity[t] <= P_battery_max * alpha_p_factor[scen, t]
            )
        
        model.addConstr(
            P_dch[scen, t]+ P_fcr_capacity[t] <= P_battery_max * alpha_p_factor[scen, t]
            )
        model.addConstr(
            P_grid_import[scen, t] <= P_grid_max * z_grid[scen, t],
            name = f"selection3-{t}"
            )
        
        model.addConstr(
            P_grid_export[scen, t] <= P_grid_max * (1-z_grid[scen, t]),
            name = f"selection4-{t}"
            )
        
        model.addConstr(
            E_imbalance_positive[scen, t] <= z_imbalance_positive[scen, t] * (E_bid_max_DA + E_bid_max_ID),
            name = "E_imb_pos_constr"
            )
        model.addConstr(
            E_imbalance_negative[scen, t] <= z_imbalance_negative[scen, t] * (E_bid_max_DA + E_bid_max_ID),
            name = "E_imb_neg_constr"
            )  
        
        model.addConstr(
            z_imbalance_positive[scen, t] +z_imbalance_negative[scen, t] <= 1,
            name = "Imbalance pos neg selection"
            )
        
        
        model.addConstr(
            P_grid_import[scen, t] + P_fcr_capacity[t] <= P_grid_max 
            )
        
        model.addConstr(
            P_grid_export[scen, t] + P_fcr_capacity[t] <= P_grid_max 
            )
        
        model.addConstr(
            P_flex_up[scen, t] <= p_flex_cap_up[t] * z_flex_up[scen, t], 
            name = f"flexUP{t}"
            )
        
        model.addConstr(
            P_flex_down[scen, t] <= p_flex_cap_down[t] * z_flex_down[scen, t], 
            name = f"flexDOWN{t}"
            )
        
        model.addConstr( 
            z_flex_up[scen, t] + z_flex_down[scen, t] <= 1,
            name=f"no simultaneous up and down flex request {t}"
            )
        
        model.addConstr(
            E[scen, t+1] == E[scen, t] + P_ch[scen, t] * efficiency_ch - P_dch[scen, t] * (1/efficiency_dch), 
            name=f"Energy_state{t}"
            )
        
        model.addConstr(
            E[scen, t] + P_fcr_capacity[t] * efficiency_ch * h_activation <= E_max_fixed,
            name = f"E_fcr_buffer_max{t}"
            )
        
        model.addConstr(
            E[scen, t] - P_fcr_capacity[t] * (1 / efficiency_dch) * h_activation >= E_min_fixed,
            name = f"E_fcr_buffer_min{t}"
            )
            
        model.addConstr(
            P_ch[scen, t] + p_load_real[t] + P_grid_import[scen, t] + P_flex_up[scen, t] + P_comp_up[scen, t]
            - P_flex_down[scen, t] - P_comp_down[scen, t] - P_grid_export[scen, t] - P_dch[scen, t] - p_pv_scenarios[scen] [t] == 0,
            name=f"power_balance{t}"
            )
        
        model.addConstr(
            (P_grid_import[scen, t] - P_grid_export[scen, t]) * delta_t == 
            E_buy_DA[t] - E_sell_DA[t] + 
            E_buy_ID[scen, t] - E_sell_ID[scen, t] +
            E_imbalance_positive[scen, t] - E_imbalance_negative[scen, t],
            name= f"buy_sell_imbalance{t}"
            )
        
        model.addConstr(
            P_grid_export[scen, t] <= p_grid_export_cap[t]
            )
        
        model.addConstr(
            gp.quicksum(lambda_p_soc[scen, t,k] for k in range(N_SOC)) == 1,
            name=f"lambda_set{t}"
            )
        
        model.addConstr(
            E[scen, t] == gp.quicksum(lambda_p_soc[scen, t, k] * E_max_fixed * soc_set[k] 
                                for k in range(N_SOC)),
            name=f"SoC_interpolate_{t}"
            )
        
        model.addConstr(
            alpha_p_factor[scen, t] == gp.quicksum(lambda_p_soc[scen, t, k] * alpha_p_set[k] 
                                              for k in range(N_SOC)),
            name=f"Alpha_interpolate_{t}"
        )
        
        model.addSOS(
            GRB.SOS_TYPE2, 
            [lambda_p_soc[scen, t, k] for k in range(N_SOC)]
            )
        
        model.addConstr(
            P_ch[scen, t] <= P_battery_max * alpha_p_factor[scen, t],
            name=f"Power_Limit_SoC_ch_{t}"
        )
    
        # Repeat for discharge (P_dch)
        model.addConstr(
            P_dch[scen, t] <= P_battery_max * alpha_p_factor[scen, t],
            name=f"Power_Limit_SoC_dch_{t}"
        )
    
        
        
        if (t+1)%24==0:
            model.addConstr(
                E[scen, t]>=2*E_min_fixed,
                name="Energy_final"
                )

for scen in scenarios:
    #No down flex for more than 3 hours, 3 selected to avoid initial condition conflict
    model.addConstrs(
        (z_flex_down[scen, t]+z_flex_down[scen, t+3] <= 1 for t in range(T-3)),
        name="no two downs"
        )
    
    #No up flex for more than 3 hour
    model.addConstrs(
        (z_flex_up[scen, t]+z_flex_up[scen, t+3] <= 1 for t in range(T-3)),
        name="no two ups"
        )
    
    
    
    # Rebound compensation for the flexibility
    for t in range(0, T-flex_comp_delay):
        model.addConstr(
            P_comp_up[scen, t+flex_comp_delay] == P_flex_down[scen, t],
            name="P_comp_up"
            )
        model.addConstr(
            P_comp_down[scen, t+flex_comp_delay] == P_flex_up[scen, t],
            name="P_comp_down"
            )
    
    # Block rebound comp in the first "flex_comp_delay" hours 
    for t in range(flex_comp_delay):
        model.addConstr(
            P_comp_up[scen, t] == 0
            )
        
        model.addConstr(
            P_comp_down[scen, t]== 0
            )
    # Block flexibility activation in the last "flex_comp_delay" hours 
    # if you dont want comp. debt in the next run
    
    for t in range(T-flex_comp_delay, T):
        model.addConstr(
            P_flex_up[scen, t] == 0, 
            name=f"no_up_near_end{t}"
            )
        
        model.addConstr(
            P_flex_down[scen, t] == 0, 
            name=f"no_down_near_end{t}"
            )
        
        model.addConstr(
            z_flex_up[scen, t] == 0, 
            name=f"z_no_up_near_end{t}"
            )
        
        model.addConstr(
            z_flex_down[scen, t] == 0, 
            name=f"z_no_down_near_end[{t}]"
            )
    
    #Constraint on max cumulative flex requested    
    model.addConstr(
        gp.quicksum(P_flex_down[scen, t] + P_flex_up[scen, t] for t in range(T)) <= 
        flex_share_of_load_max * sum(p_load_real)
        )


  
#Obj Function+ Opt.=======================================

#Set Obj
penalty_flex = 1 # penalty factor, can be tailored to per hour based on the fex requirement
# for flex, everything does not need to be a penalty, like in case of grid congestion with surplus power, flex Up can be incentivised 

#maximise profit with flex penalty
r = gp.quicksum(
    pi_sell_DA[t] * E_sell_DA[t] - pi_buy_DA[t] * E_buy_DA[t] +
    probability * ( # Weight the scenario
        pi_sell_ID[t] * E_sell_ID[scen, t] 
        - pi_buy_ID[t] * E_buy_ID[scen, t] 
        - penalty_positive_imbalance[t] * E_imbalance_positive[scen, t]
        - penalty_negative_imbalance[t] * E_imbalance_negative[scen, t]
        - penalty_flex * (P_comp_down[scen, t] + P_comp_up[scen, t])
        - pi_battery_degradation * (P_ch[scen, t] + P_dch[scen, t])
    )
    for scen in scenarios for t in range(T) # Iterate over scenarios AND time
)

model.setObjective(r, GRB.MAXIMIZE)

#Run the optimization
model.setParam('DualReductions', 0) # for debugging
model.optimize()


#Results Plot=======================================================

if model.status == GRB.OPTIMAL:
    
    # --- 1. Extract Data ---
    
    # DA Bids (Deterministic / First Stage) - 1D Array
    da_buy_vals = np.array([E_buy_DA[t].X for t in time_steps])
    da_sell_vals = np.array([E_sell_DA[t].X for t in time_steps])

    # ID Bids (Stochastic / Second Stage) - 2D Array [Scenario][Time]
    id_buy_vals = np.zeros((no_of_scenarios, T))
    id_sell_vals = np.zeros((no_of_scenarios, T))

    for s in scenarios:
        for t in time_steps:
            id_buy_vals[s, t] = E_buy_ID[s, t].X
            id_sell_vals[s, t] = E_sell_ID[s, t].X

    # --- 2. Plotting ---
    
    plt.figure(figsize=(14, 7))
    
    # A. Plot Intraday (ID) Scenarios
    # We iterate through scenarios to plot "spaghetti" lines
    for s in scenarios:
        # Use label=None for s > 0 to avoid cluttering the legend
        lbl_buy = 'ID Buy (Scenarios)' if s == 0 else "_nolegend_"
        lbl_sell = 'ID Sell (Scenarios)' if s == 0 else "_nolegend_"
        
        plt.step(time_steps, id_buy_vals[s, :], where='mid', 
                 color='deepskyblue', alpha=0.3, linewidth=1, linestyle='--', label=lbl_buy)
        
        # Plot Sell as negative for easier visual comparison? 
        # Or positive with different color. Here we plot positive to compare magnitudes.
        plt.step(time_steps, id_sell_vals[s, :], where='mid', 
                 color='salmon', alpha=0.3, linewidth=1, linestyle='--', label=lbl_sell)

    # B. Plot Day-Ahead (DA) Bids
    # Plot these last so they appear on top (zorder)
    plt.step(time_steps, da_buy_vals, where='mid', 
             color='blue', linewidth=2.5, label='DA Buy (Fixed)')
             
    plt.step(time_steps, da_sell_vals, where='mid', 
             color='red', linewidth=2.5, label='DA Sell (Fixed)')

    # --- 3. Styling ---
    plt.title(f'Market Bidding Strategy: First Stage (DA) vs Second Stage Recourse (ID)\n({no_of_scenarios} Scenarios)', fontsize=14)
    plt.ylabel('Energy Bid (MWh)', fontsize=12)
    plt.xlabel('Hour (t)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    # Optional: Highlight grid export restricted zones if relevant
    for d in range(no_of_days):
        start = d*24 + 10 # Adjust based on your p_grid_export_cap logic
        end = d*24 + 14
        plt.axvspan(start, end, color='gray', alpha=0.1, label='Grid Constraint' if d==0 else "")

    plt.tight_layout()
    plt.show()

else:
    print("Model did not solve to optimality.")


