import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Read data 
current_script_path = os.path.abspath(__file__)
base_dir = os.path.dirname(current_script_path)

conventional_generators = os.path.join(base_dir, "conventional_generators.txt")
wind_farms = os.path.join(base_dir, "wind_farms.txt")
demands = os.path.join(base_dir, "demands.txt")

generators = pd.read_csv(conventional_generators, sep = ',')
wind_farms = pd.read_csv(wind_farms, sep = ',')  
demands = pd.read_csv(demands, sep = ',')

wind_farms['prod_cost_per_MWh'] = 0.0 
wind_farms.rename(columns={"day_ahead_forecast_MW": "capacity_MW"}, inplace=True)
demands["bid_price_per_MWh"] = np.linspace(500, 0, len(demands))

HOURS = list(range(24))

load_profile = np.array([0.6, 0.55, 0.5, 0.5, 0.55, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0, 0.95,
                         0.9, 0.85, 0.85, 0.85, 0.9, 1.0, 0.95, 0.9, 0.8, 0.7, 0.65, 0.6])
wind_profile = np.array([0.8, 0.85, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2,
                         0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.85, 0.8])

ES_PARAMS = {'cap': 600.0, 'power': 150.0, 'eta_ch': 0.8, 'eta_dis': 0.9, 'soc_ini': 300.0}

# ===================================================================
# 2. Packaging the Solver Function
# ===================================================================
def solve_market(use_storage=True):
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
        # 1. Generation & Wind Limits
        for g in generators.index:
            model.addConstr(pg[g, t] <= generators.loc[g, "capacity_MW"])
        for w in wind_farms.index:
            model.addConstr(pw[w, t] <= wind_farms.loc[w, "capacity_MW"] * wind_profile[t])
        
        # 2. Demand Limits
        for d in demands.index:
            d_val = demands.loc[d, "consumption_MW"] * load_profile[t]
            model.addConstr(pd_served[d, t] <= d_val)

        # 3. Storage Constraints
        if use_storage:
            model.addConstr(p_ch[t] <= ES_PARAMS['power'])
            model.addConstr(p_dis[t] <= ES_PARAMS['power'])
            model.addConstr(soc[t] <= ES_PARAMS['cap'])
            
            prev_soc = ES_PARAMS['soc_ini'] if t == 0 else soc[t-1]
            model.addConstr(soc[t] == prev_soc + p_ch[t]*ES_PARAMS['eta_ch'] - p_dis[t]/ES_PARAMS['eta_dis'])
        else:
            model.addConstr(p_ch[t] == 0)
            model.addConstr(p_dis[t] == 0)
            model.addConstr(soc[t] == 0)

        # 4. Power Balance
        supply = gp.quicksum(pg[g, t] for g in generators.index) + \
                 gp.quicksum(pw[w, t] for w in wind_farms.index) + p_dis[t]
        demand = gp.quicksum(pd_served[d, t] for d in demands.index) + p_ch[t]
        
        # Save constraint to extract shadow price later
        balance_constrs[t] = model.addConstr(supply == demand)

    if use_storage:
        model.addConstr(soc[HOURS[-1]] >= ES_PARAMS['soc_ini'])

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
        net_storage = [p_dis[t].X - p_ch[t].X for t in HOURS]
        soc_level = [soc[t].X for t in HOURS]

        # Generator profits
        gen_profits = []
        for g in generators.index:
            p_total = sum(pg[g, t].X for t in HOURS)
            rev = sum(pg[g, t].X * prices[t] for t in HOURS) 
            cst = p_total * generators.loc[g, "prod_cost_per_MWh"]
            gen_profits.append({'id': g, 'Produced_MW': p_total, 'Profit': abs(rev - cst)})
        
        total_op_cost = sum(pg[g, t].X * generators.loc[g, "prod_cost_per_MWh"] 
                            for g in generators.index for t in HOURS)

        # Profit of EACH producer
        all_producers_profit = []
        # A. conventional
        for g in generators.index:
            p_total = sum(pg[g, t].X for t in HOURS)
            rev = sum(pg[g, t].X * prices[t] for t in HOURS)
            cst = p_total * generators.loc[g, "prod_cost_per_MWh"]
            g_name = generators.iloc[g, 0] if isinstance(generators.iloc[g, 0], str) else f"Gen_{g}"
            all_producers_profit.append({
                'Name': g_name, 'Type': 'Thermal', 'Total_Prod_MWh': p_total,
                'Revenue': rev, 'Cost': cst, 'Profit': abs(rev - cst)
            })

        # B. Wind
        for w in wind_farms.index:
            p_total = sum(pw[w, t].X for t in HOURS)
            rev = sum(pw[w, t].X * prices[t] for t in HOURS)
            cst = 0.0 
            w_name = wind_farms.iloc[w, 0] if isinstance(wind_farms.iloc[w, 0], str) else f"Wind_{w}"
            all_producers_profit.append({
                'Name': w_name, 'Type': 'Wind', 'Total_Prod_MWh': p_total,
                'Revenue': rev, 'Cost': cst, 'Profit': abs(rev - cst)
            })
            
        # Storage profit
        es_rev = sum(p_dis[t].X * prices[t] for t in HOURS)
        es_cost = sum(p_ch[t].X * prices[t] for t in HOURS)
        es_profit = abs(es_rev - es_cost)

        return prices, gen_total, net_storage, soc_level, pd.DataFrame(gen_profits), pd.DataFrame(all_producers_profit), es_profit, model.ObjVal, total_op_cost
    else:
        return None

# ===================================================================
# 3. Output & Display
# ===================================================================

print("Now calculating (No Storage)...")
res_no = solve_market(use_storage=False)

print("Now calculating (With Storage)...")
res_yes = solve_market(use_storage=True)

if res_no and res_yes:
    price_no, gen_no, _, _, gen_profits_no, df_profit_no, _, sw_no, cost_no = res_no
    price_yes, gen_yes, net_store_yes, soc_yes, gen_profits_yes, df_profit_yes, es_profit_yes, sw_yes, cost_yes = res_yes
    
    # --- Output 1: System summary table ---
    print("\n" + "="*50)
    print("Table 1: System Economics Summary (All Absolute Values)")
    print("="*50)
    df_summary = pd.DataFrame({
        'Metric': ['Total Social Welfare (€)', 'Avg. Price (€/MWh)', 'Storage Profit (€)', 'Price Std Dev'],
        'No Storage': [sw_no, np.mean(price_no), 0.0, np.std(price_no)],
        'With Storage': [sw_yes, np.mean(price_yes), es_profit_yes, np.std(price_yes)]
    })
    df_summary['Diff'] = (df_summary['With Storage'] - df_summary['No Storage']).abs()
    print(df_summary.round(2).to_string(index=False))

    # --- Output 2: Generator Profits Analysis ---
    print("\n" + "="*50)
    print("Table 2: Generator Profits Analysis")
    print("="*50)
    df_gen = gen_profits_no[['id', 'Profit']].rename(columns={'Profit': 'Profit_NoES'})
    df_gen['Profit_WithES'] = gen_profits_yes['Profit']
    
    # Magnitude of change
    df_gen['Change'] = (df_gen['Profit_WithES'] - df_gen['Profit_NoES']).abs()
    df_gen['MC'] = generators['prod_cost_per_MWh']
    print(df_gen.round(2).to_string(index=False))
    
    # --- Output 3: system cost ---
    print("\n" + "="*50)
    print("Table 3: Total Operating Cost (24h)")
    print("="*50)
    print(f"  No Storage System Cost:   €{cost_no:,.2f}")
    print(f"  With Storage System Cost: €{cost_yes:,.2f}")
    print(f"  Cost Saving:              €{abs(cost_no - cost_yes):,.2f}")

    # --- Output 4: profit of each producer ---
    print("\n" + "="*80)
    print("Table 4: Total Profit of Each Producer (24h) - With Storage Scenario")
    print("="*80)
    cols = ['Name', 'Type', 'Total_Prod_MWh', 'Revenue', 'Cost', 'Profit']
    print(df_profit_yes[cols].round(2).to_string(index=False))

    print("\n" + "="*80)
    print("Compare Total Profit (Wind vs ThermSal) - With Storage")
    print("="*80)
    summary_profit = df_profit_yes.groupby('Type')['Profit'].sum().reset_index()
    print(summary_profit.round(2).to_string(index=False))

    # --- Output 5: Price Comparison ---
    print("\n" + "="*50)
    print("Table 5: Price Comparison")
    print("="*50)
    print(f"{'Hour':<6} {'Price(No ES)':<15} {'Price(With ES)':<15} {'|Diff|':<10}")
    print("-" * 45)
    for t in HOURS:
        diff = abs(price_yes[t] - price_no[t])
        print(f"{t+1:<6d} {price_no[t]:<15.2f} {price_yes[t]:<15.2f} {diff:<10.2f}")
    print("="*45)

    # ===================================================================
    # 4. Visualization
    # ===================================================================
    
    # --- fig 1: Price Comparison ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(HOURS, price_no, 'r--o', label='Price (No Storage)', alpha=0.7)
    ax1.plot(HOURS, price_yes, 'b-o', label='Price (With Storage)', linewidth=2)
    ax1.set_ylabel('Price (€/MWh)')
    ax1.set_xlabel('Hour of Day (0-23)') 
    ax1.set_title('Impact of Storage on Electricity Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- fig 2: Storage Dispatch & SoC ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    net_storage_vals = np.array(net_store_yes)
    colors = ['green' if x >= 0 else 'red' for x in net_storage_vals]
    ax2.bar(HOURS, np.abs(net_storage_vals), color=colors, alpha=0.6, label='Net Output Magnitude')
    ax2.axhline(0, color='black', linewidth=0.8)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(HOURS, soc_yes, 'k:', label='SoC Level', linewidth=2)
    ax2_twin.set_ylabel('SoC (MWh)')
    ax2_twin.set_ylim(0, ES_PARAMS['cap'] * 1.1)
    
    ax2.set_xlabel('Hour of Day (0-23)')
    ax2.set_ylabel('Power Magnitude (MW)\n(Green=Discharge, Red=Charge)')
    ax2.set_title('Storage Dispatch (Absolute) & SoC Profile')
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- fig 3: Thermal Gen ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(HOURS, gen_no, 'r--', label='Thermal Gen (No Storage)')
    ax3.plot(HOURS, gen_yes, 'b-', label='Thermal Gen (With Storage)')
    ax3.fill_between(HOURS, gen_no, gen_yes, where=(np.array(gen_no)>np.array(gen_yes)), 
                     interpolate=True, color='green', alpha=0.2, label='Saved Generation')
    ax3.set_ylabel('Total Thermal Gen (MW)')
    ax3.set_xlabel('Hour of Day (0-23)')
    ax3.set_title('Impact on Conventional Generation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
else:
    print("Optimization failed.")
