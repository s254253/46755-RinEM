import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================================================================
# 1. Read data & Initialization
# ===================================================================
current_script_path = os.path.abspath(__file__)
base_dir = os.path.dirname(current_script_path)

conventional_generators = os.path.join(base_dir, "conventional_generators.txt")
wind_farms = os.path.join(base_dir, "wind_farms.txt")
system_demand_file = os.path.join(base_dir, "system_demand.csv")
node_distribution_file = os.path.join(base_dir, "node_distribution.csv")

generators = pd.read_csv(conventional_generators, sep = ',')
wind_farms = pd.read_csv(wind_farms, sep = ',')  

demands = pd.read_csv(node_distribution_file, sep = ',')
demands['Share'] = demands['Percent_of_System_Load'] / 100.0

sys_demand_df = pd.read_csv(system_demand_file, sep = ',')
system_demand_MW = sys_demand_df['System_Demand_MW'].values

wind_farms['prod_cost_per_MWh'] = 0.0 
wind_farms.rename(columns={"day_ahead_forecast_MW": "capacity_MW"}, inplace=True)
demands['bid_price_per_MWh'] = (
    (np.exp(-np.linspace(0, 3.5, len(demands))) - np.exp(-6))
    / (1 - np.exp(-6))
    * 200
)
HOURS = list(range(24))

wind_profile = np.array([0.5293, 0.5927, 0.68, 0.7384, 0.7667, 0.7739, 0.7787, 0.7861, 0.7779, 0.7713,
 0.7493, 0.7033, 0.6722, 0.652, 0.6368, 0.6462, 0.6351, 0.6358, 0.6582, 0.6725,
 0.6802, 0.7034, 0.6833, 0.6896])

# define energy storage parameters for the scenarios
ES_PARAMS_DEFAULT = {'cap': 100.0, 'power': 150.0, 'eta_ch': 0.9, 'eta_dis': 0.97, 'soc_ini': 0.0}
ES_PARAMS_POOR = {'cap': 100.0, 'power': 150.0, 'eta_ch': 0.7, 'eta_dis': 0.8, 'soc_ini': 0.0} # 修改为较差状态

# ===================================================================
# 2. Packaging the Solver Function
# ===================================================================
def solve_market(use_storage=True, wind_mult=1.0, es_params=ES_PARAMS_DEFAULT):
    model = gp.Model("Market_Clearing")
    model.setParam('OutputFlag', 0)

    # --- variables ---
    pg = model.addVars(generators.index, HOURS, lb=0, name="pg")
    pw = model.addVars(wind_farms.index, HOURS, lb=0, name="pw")
    pd_served = model.addVars(demands.index, HOURS, lb=0, name="pd_served")
    
    p_ch = model.addVars(HOURS, lb=0, name="Charge")
    p_dis = model.addVars(HOURS, lb=0, name="Discharge")
    soc = model.addVars(HOURS, lb=0, name="SoC")

    # --- constraints ---
    balance_constrs = {}

    for t in HOURS:
        # 1. Generation & Wind Limits (Applied Wind Multiplier here)
        for g in generators.index:
            model.addConstr(pg[g, t] <= generators.loc[g, "capacity_MW"])
        for w in wind_farms.index:
            model.addConstr(pw[w, t] <= wind_farms.loc[w, "capacity_MW"] * wind_profile[t] * wind_mult)
        
        # 2. Demand Limits
        for d in demands.index:
            d_val = system_demand_MW[t] * demands.loc[d, "Share"]
            model.addConstr(pd_served[d, t] <= d_val)

        # 3. Storage Constraints
        if use_storage:
            model.addConstr(p_ch[t] <= es_params['power'])
            model.addConstr(p_dis[t] <= es_params['power'])
            model.addConstr(soc[t] <= es_params['cap'])
            
            prev_soc = es_params['soc_ini'] if t == 0 else soc[t-1]
            model.addConstr(soc[t] == prev_soc + p_ch[t]*es_params['eta_ch'] - p_dis[t]/es_params['eta_dis'])
        else:
            model.addConstr(p_ch[t] == 0)
            model.addConstr(p_dis[t] == 0)
            model.addConstr(soc[t] == 0)

        # 4. Power Balance
        supply = gp.quicksum(pg[g, t] for g in generators.index) + \
                 gp.quicksum(pw[w, t] for w in wind_farms.index) + p_dis[t]
        demand = gp.quicksum(pd_served[d, t] for d in demands.index) + p_ch[t]
        
        balance_constrs[t] = model.addConstr(supply == demand)

    # --- objective function ---
    obj = 0
    for t in HOURS:
        util = gp.quicksum(pd_served[d, t] * demands.loc[d, "bid_price_per_MWh"] for d in demands.index)
        cost_gen = gp.quicksum(pg[g, t] * generators.loc[g, "prod_cost_per_MWh"] for g in generators.index)
        obj += (util - cost_gen)

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    # --- Extract Results ---
    if model.status == GRB.OPTIMAL:
        prices = [abs(balance_constrs[t].Pi) for t in HOURS]
        gen_total = [sum(pg[g, t].X for g in generators.index) for t in HOURS]
        wind_total = [sum(pw[w, t].X for w in wind_farms.index) for t in HOURS]
        
        # Calculate daily revenue for the scenario
        total_revenue = 0
        for g in generators.index:
            total_revenue += sum(pg[g, t].X * prices[t] for t in HOURS)
        for w in wind_farms.index:
            total_revenue += sum(pw[w, t].X * prices[t] for t in HOURS)
            
        return {
            'prices': prices,
            'gen_total': gen_total,
            'wind_total': wind_total,
            'total_revenue': total_revenue
        }
    else:
        print("Optimization failed.")
        return None

# ===================================================================
# 3. Run Scenarios for Sensitivity Analysis
# ===================================================================
print("Running Sensitivity Analysis Scenarios...")

scenarios = {
    'W/O battery':      solve_market(use_storage=False, wind_mult=1.0, es_params=ES_PARAMS_DEFAULT),
    'W battery':        solve_market(use_storage=True,  wind_mult=1.0, es_params=ES_PARAMS_DEFAULT),
    'High Wind':        solve_market(use_storage=True,  wind_mult=6.0, es_params=ES_PARAMS_DEFAULT),
    'Poor Battery':     solve_market(use_storage=True,  wind_mult=1.0, es_params=ES_PARAMS_POOR)
}

# ===================================================================
# 4. Visualization (Mimicking the reference report style)
# ===================================================================

# -------------------------------------------------------------------
# Figure 1: (a) Wind power & (b) Total power generation
# -------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot (a): Wind power generation compared to demand ---
ax_wind = axes1[0]
total_wind_cap = wind_farms['capacity_MW'].sum()

for name in scenarios.keys():
    mult = 6.0 if name == 'High Wind' else 1.0
    wind_generation = total_wind_cap * wind_profile * mult
    
    # adjust line styles and colors to mimic the reference report
    if name == 'W/O battery':
        ax_wind.step(HOURS, wind_generation, where='post', label=name, color='k', linestyle='-', linewidth=6, alpha=0.2)
    elif name == 'W battery':
        ax_wind.step(HOURS, wind_generation, where='post', label=name, color='g', linestyle='--', linewidth=2)
    elif name == 'Poor Battery':
        ax_wind.step(HOURS, wind_generation, where='post', label=name, color='b', linestyle=':', linewidth=2.5)
    else: # High Wind
        ax_wind.step(HOURS, wind_generation, where='post', label=name, color='r', linestyle='-', linewidth=1.5)

ax_wind.step(HOURS, system_demand_MW, where='post', label='Demand', color='m', linestyle='-.', linewidth=1.5)
ax_wind.set_xlabel('Time [h]')
ax_wind.set_ylabel('MW')
ax_wind.set_title('(a) Wind power generation compared to demand')
ax_wind.legend(title='Scenario')
ax_wind.grid(True, alpha=0.3)

# --- Plot (b): Total power generated (Gen + Wind) compared to demand ---
ax_gen = axes1[1]
for name, res in scenarios.items():
    if res:
        total_gen = np.array(res['gen_total']) + np.array(res['wind_total'])
        if name == 'W/O battery':
            ax_gen.step(HOURS, total_gen, where='post', label=name, color='k', linestyle='-', linewidth=6, alpha=0.2)
        elif name == 'W battery':
            ax_gen.step(HOURS, total_gen, where='post', label=name, color='g', linestyle='--', linewidth=2)
        elif name == 'Poor Battery':
            ax_gen.step(HOURS, total_gen, where='post', label=name, color='b', linestyle=':', linewidth=2.5)
        else: # High Wind
            ax_gen.step(HOURS, total_gen, where='post', label=name, color='r', linestyle='-', linewidth=1.5)

ax_gen.step(HOURS, system_demand_MW, where='post', label='Demand', color='m', linestyle='-.', linewidth=1.5)
ax_gen.set_xlabel('Time [h]')
ax_gen.set_ylabel('MW')
ax_gen.set_title('(b) Total power generation compared to demand')
ax_gen.legend(title='Scenario')
ax_gen.grid(True, alpha=0.3)

fig1.tight_layout()
plt.show() # show the first figure before plotting the second one

# -------------------------------------------------------------------
# Figure 2: (c) Market clearing price & (d) Daily Revenue
# -------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot (c): Market clearing price ---
ax_price = axes2[0]
for name, res in scenarios.items():
    if res:
        if name == 'W/O battery':
            # set the black line with low alpha to mimic the reference report style, allowing other lines to be visible on top
            ax_price.step(HOURS, res['prices'], where='post', label=name, color='k', linestyle='-', linewidth=6, alpha=0.2)
        elif name == 'W battery':
            ax_price.step(HOURS, res['prices'], where='post', label=name, color='g', linestyle='--', linewidth=2)
        elif name == 'Poor Battery':
            # set the blue line with a dotted style and slightly thicker width to mimic the reference report style, making it visually distinct
            ax_price.step(HOURS, res['prices'], where='post', label=name, color='b', linestyle=':', linewidth=2.5)
        else: # High Wind
            ax_price.step(HOURS, res['prices'], where='post', label=name, color='r', linestyle='-', linewidth=1.5)

ax_price.set_xlabel('Time [h]')
ax_price.set_ylabel('€/MWh')
ax_price.set_title('(c) Market clearing price of the different scenarios')
ax_price.legend()
ax_price.grid(True, alpha=0.3)

# --- Plot (d): Daily Revenue for each scenario ---
ax_rev = axes2[1]
revenues = [res['total_revenue'] for name, res in scenarios.items() if res]
scenario_names = [name for name in scenarios.keys() if scenarios[name]]

bars = ax_rev.bar(scenario_names, revenues, color='#1f77b4', edgecolor='black')
ax_rev.set_ylabel('€/day')
ax_rev.set_title('(d) Total daily revenue for each scenario')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    ax_rev.text(bar.get_x() + bar.get_width()/2, yval + (max(revenues)*0.01), 
                f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

fig2.tight_layout()
plt.show() # show the second figure after plotting the first one
