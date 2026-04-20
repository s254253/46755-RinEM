import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import matplotlib.pyplot as plt
from task1_1 import run_task_1_1

# Load results from Task 1.1 for comparison
optimal_bids_11 = run_task_1_1()

# ==========================================
# 1. Load 1600 Scenarios
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'final_1600_scenarios_input.csv'
df_path = os.path.join(script_dir, file_name)

df = pd.read_csv(df_path)
hours = df['Hour'].unique()
scenarios = df['Scenario_ID'].unique()
P_max = 500

# Extract parameter dictionaries
prob = df[['Scenario_ID', 'Probability']].drop_duplicates().set_index('Scenario_ID')['Probability'].to_dict()
DA_price = df.set_index(['Hour', 'Scenario_ID'])['DA_Price'].to_dict()
BP_price = df.set_index(['Hour', 'Scenario_ID'])['Bal_Price'].to_dict()
Wind_MW = df.set_index(['Hour', 'Scenario_ID'])['Wind_MW'].to_dict()

# 2. Build the Model
m2 = gp.Model("Task1_2_Two_Price_Scheme")

# Decision Variables
P_DA = m2.addVars(hours, lb=0, ub=P_max, name="P_DA")

# Auxiliary variables: Imbalance quantities (Positive and Negative)
delta_plus = m2.addVars(hours, scenarios, lb=0, name="delta_plus")
delta_minus = m2.addVars(hours, scenarios, lb=0, name="delta_minus")

# Constraint: Wind - P_DA = delta_plus - delta_minus
m2.addConstrs(
    (Wind_MW[t, w] - P_DA[t] == delta_plus[t, w] - delta_minus[t, w]
     for t in hours for w in scenarios), name="Imbalance_Balance"
)

# Objective Function: Maximize Expected Profit
# Settlement logic: 
# Positive imbalance (surplus) is settled at min(DA, BP)
# Negative imbalance (shortage) is settled at max(DA, BP)
obj = gp.quicksum(
    prob[w] * (
        DA_price[t, w] * P_DA[t] + 
        min(DA_price[t, w], BP_price[t, w]) * delta_plus[t, w] - 
        max(DA_price[t, w], BP_price[t, w]) * delta_minus[t, w]
    )
    for t in hours for w in scenarios
)

m2.setObjective(obj, GRB.MAXIMIZE)

# 3. Solve the Model
m2.optimize()

# 4. Results Visualization
if m2.status == GRB.OPTIMAL:
    # 1. Calculate Optimal Expected Profit
    print(f"\nTask 1.2 Optimal Expected Profit: {m2.ObjVal:,.2f}")
    results_1_2 = []
    for t in hours:
        results_1_2.append({'Hour': t, 'P_DA_1.2': P_DA[t].X})
    
    df_results = pd.DataFrame(results_1_2)
    print("\nOptimal Offers (Two-Price Scheme):")
    print(df_results.to_string(index=False))

    optimal_bids_12 = {t: P_DA[t].X for t in hours}

    # 1. Calculate detailed profit and imbalance for each scenario
    scenario_data = []
    for w in scenarios:
        total_profit_w = 0
        total_pos_imb = 0 # Surplus
        total_neg_imb = 0 # Shortage
        
        for t in hours:
            da_p = DA_price[t, w]
            bp_p = BP_price[t, w]
            wind = Wind_MW[t, w]
            bid = optimal_bids_12[t]
            
            # Settlement Logic
            imbalance = wind - bid
            if imbalance > 0: # Surplus
                profit = da_p * bid + min(da_p, bp_p) * imbalance
                total_pos_imb += imbalance
            else: # Shortage
                profit = da_p * bid + max(da_p, bp_p) * imbalance # imbalance is negative here
                total_neg_imb += abs(imbalance)
            
            total_profit_w += profit
            
        scenario_data.append({
            'Scenario_ID': w,
            'Profit': total_profit_w,
            'Pos_Imbalance': total_pos_imb,
            'Neg_Imbalance': total_neg_imb,
            'Prob': prob[w]
        })

    df_scen_analysis = pd.DataFrame(scenario_data)

    # 2. Plot Profit Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df_scen_analysis['Profit'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
    plt.axvline(m2.ObjVal, color='blue', linestyle='dashed', linewidth=2, label=f'Expected Profit: {m2.ObjVal:,.2f}')
    plt.title('Profit Distribution (Task 1.2: Two-Price Scheme)')
    plt.xlabel('Profit (€)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    #plt.savefig('profit_distribution_1_2.png')
    plt.show()

    # 3. Output Comparison and Analysis Table
    # Calculate the average hourly imbalance
    hourly_analysis = []
    for t in hours:
        bid_12 = optimal_bids_12[t]
        # Assuming Task 1.1 results were stored in optimal_bids_11
        bid_11 = optimal_bids_11[t] 
        
        avg_pos = sum(prob[w] * max(0, Wind_MW[t, w] - bid_12) for w in scenarios)
        avg_neg = sum(prob[w] * max(0, bid_12 - Wind_MW[t, w]) for w in scenarios)
        
        hourly_analysis.append({
            'Hour': t,
            'Bid_1.1 (One-Price)': bid_11,
            'Bid_1.2 (Two-Price)': bid_12,
            'Exp_Surplus_MW': avg_pos,
            'Exp_Shortage_MW': avg_neg
        })

    df_hourly_comp = pd.DataFrame(hourly_analysis)
    print("\n" + "="*80)
    print("Task 1.1 vs Task 1.2 Comparison and Imbalance Analysis")
    print("="*80)
    print(df_hourly_comp.to_string(index=False, formatters={'Bid_1.2 (Two-Price)': '{:.2f}'.format, 
                                                        'Exp_Surplus_MW': '{:.2f}'.format,
                                                        'Exp_Shortage_MW': '{:.2f}'.format}))