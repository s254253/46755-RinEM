import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Load 1600 Scenarios
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'final_1600_scenarios_input.csv'
df_path = os.path.abspath(
    os.path.join(script_dir, '..', 'scenario_prep', file_name)
)

def run_task_1_1():
    df = pd.read_csv(df_path)

    # Extract basic sets and parameters
    hours = df['Hour'].unique()
    scenarios = df['Scenario_ID'].unique()
    P_max = 500  # Wind farm installed capacity: 500 MW

    # Convert DataFrame to dictionaries for efficient constraint and objective construction in Gurobi
    # Extract probabilities (assuming fixed probability for each Scenario_ID)
    prob = df[['Scenario_ID', 'Probability']].drop_duplicates().set_index('Scenario_ID')['Probability'].to_dict()

    # Extract prices and wind power output: dict[(hour, scenario)] = value
    DA_price = df.set_index(['Hour', 'Scenario_ID'])['DA_Price'].to_dict()
    Bal_price = df.set_index(['Hour', 'Scenario_ID'])['Bal_Price'].to_dict()
    Wind_MW = df.set_index(['Hour', 'Scenario_ID'])['Wind_MW'].to_dict()

    # 2. Build Gurobi Optimization Model
    m = gp.Model("Task1_1_One_Price_Scheme")

    # Define decision variables: Day-ahead market bidding quantity for 24 hours (Here-and-now variable)
    # Variable bounds are directly limited between 0 and 500 MW
    P_DA = m.addVars(hours, vtype=GRB.CONTINUOUS, lb=0, ub=P_max, name="P_DA")

    # Define objective function: Maximize expected profit
    # Expected Profit = sum_w prob[w] * sum_t [ DA[t,w]*P_DA[t] + BP[t,w]*(Wind[t,w] - P_DA[t]) ]
    expected_profit = gp.quicksum(
        prob[w] * (DA_price[t, w] * P_DA[t] + Bal_price[t, w] * (Wind_MW[t, w] - P_DA[t]))
        for t in hours for w in scenarios
    )

    m.setObjective(expected_profit, GRB.MAXIMIZE)

    # 3. Solve the Model
    m.optimize()

    # 4. Extract Results and Validation
    if m.status == GRB.OPTIMAL:
        print(f"\nOptimal Expected Total Profit: {m.ObjVal:,.2f}")
        
        print("\nOptimal Day-Ahead Bidding Quantity per Hour (MW):")
        optimal_bids = {}
        for t in hours:
            optimal_bids[t] = P_DA[t].X
            print(f"Hour {t}: {optimal_bids[t]} MW")
            
        # Calculate and collect specific total profit realized under each scenario
        scenario_profits = []
        for w in scenarios:
            profit_w = sum(
                DA_price[t, w] * optimal_bids[t] + Bal_price[t, w] * (Wind_MW[t, w] - optimal_bids[t]) 
                for t in hours
            )
            scenario_profits.append(profit_w)
            
        # 5. Plot Profit Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(scenario_profits, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(m.ObjVal, color='red', linestyle='dashed', linewidth=2, label=f'Expected Profit: {m.ObjVal:,.2f}')
        plt.title('Profit Distribution (One-Price Scheme)')
        plt.xlabel('Profit')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.7)
        #plt.savefig('profit_distribution.png')
        plt.show()

        # 6. Calculate expected hourly DA and Bal prices
        # 1. Calculate expected prices per hour (considering probability weights for each scenario)
        # Use groupby('Hour') and apply a weighted average to each group
        expected_summary = df.groupby('Hour').apply(
            lambda x: pd.Series({
                'Exp_DA_Price': np.average(x['DA_Price'], weights=x['Probability']),
                'Exp_Bal_Price': np.average(x['Bal_Price'], weights=x['Probability'])
            })
        ).reset_index()

        # 2. Calculate the Spread
        # The sign of this Spread determines whether the optimal bid is 500 or 0
        expected_summary['Price_Spread'] = expected_summary['Exp_DA_Price'] - expected_summary['Exp_Bal_Price']

        # 3. Add the optimized bidding decisions for comparison
        expected_summary['Optimal_P_DA'] = np.where(expected_summary['Price_Spread'] > 0, 500, 0)

        # 4. Print results table, formatted to two decimal places
        print("-" * 70)
        print(f"{'Hour':<6} | {'E[DA]':<10} | {'E[BP]':<10} | {'Spread':<10} | {'Offer (MW)':<10}")
        print("-" * 70)
        for index, row in expected_summary.iterrows():
            print(f"{int(row['Hour']):<6} | {row['Exp_DA_Price']:<10.2f} | {row['Exp_Bal_Price']:<10.2f} | "
                f"{row['Price_Spread']:<10.2f} | {row['Optimal_P_DA']:<10.0f}")
        print("-" * 70)
        
    else:
        print("The model failed to find an optimal solution.")
    
    optimal_bids = {}
    if m.status == GRB.OPTIMAL:
        for t in hours:
            optimal_bids[t] = P_DA[t].X

    return optimal_bids, m.objVal


if __name__ == "__main__":
    bids, obj_val = run_task_1_1()
    print("Task 1.1 completed.")
    print(f"Optimal objective value: {obj_val}")