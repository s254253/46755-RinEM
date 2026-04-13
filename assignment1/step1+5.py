import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Read data

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
df_path=os.path.join(script_dir, 'conventional_generators.txt')
generators = pd.read_csv(df_path, sep = ',')
df_path=os.path.join(script_dir, 'wind_farms.txt')
wind_farms = pd.read_csv(df_path, sep = ',')  
df_path=os.path.join(script_dir, 'demands.txt')
demands = pd.read_csv(df_path, sep = ',')


wind_farms['prod_cost_per_MWh'] = 0.0  # zero marginal cost for wind
wind_farms.rename(columns={'day_ahead_forecast_MW': 'capacity_MW'}, inplace=True)

# Demand bid prices (descending, exponential-like from high to low)
demands['bid_price_per_MWh'] = (
    (np.exp(-np.linspace(0, 3.5, len(demands))) - np.exp(-6))
    / (1 - np.exp(-6))
    * 200
)

# -------------------------------------------------------------------
# 2. Build SUPPLY curve (Merit Order)
# -------------------------------------------------------------------

supply = pd.concat([
    generators[['capacity_MW', 'prod_cost_per_MWh']],
    wind_farms[['capacity_MW', 'prod_cost_per_MWh']]
])

supply = supply.sort_values(
    by=['prod_cost_per_MWh', 'capacity_MW'],
    ascending=[True, True]
).reset_index(drop=True)

supply['cum_quantity'] = supply['capacity_MW'].cumsum()

# -------------------------------------------------------------------
# 3. Build DEMAND curve
# -------------------------------------------------------------------

demands_sorted = demands.sort_values(
    by='bid_price_per_MWh',
    ascending=False
).reset_index(drop=True)

demands_sorted['cum_quantity'] = (
                                demands_sorted['consumption_MW']
                                .cumsum()
                                .shift(fill_value=0)
)
demands_sorted.loc[demands.index[-1], 'bid_price_per_MWh'] = 0.0  # zero price for last demand point
# -------------------------------------------------------------------
# 4. Build optimization model
# -------------------------------------------------------------------

model = gp.Model('CopperPlate_MarketClearing_Step1')

# pg[g]: power produced by conventional generator g [MW]
pg = model.addVars(
    generators.id,
    lb=0,
    ub=generators.set_index('id').capacity_MW,
    name='pg'
)

# pw[w]: power produced by wind farm w [MW]
pw = model.addVars(
    wind_farms.id,
    lb=0,
    ub=wind_farms.set_index('id').capacity_MW,
    name='pw'
)

# pd[d]: served demand d [MW]
pd = model.addVars(
    demands.id,
    lb=0,
    ub=demands.set_index('id').consumption_MW,
    name='pd'
)

# -------------------------------------------------------------------
# 5. Objective function: maximize social welfare
# -------------------------------------------------------------------

model.setObjective(
    gp.quicksum(
        demands.loc[demands.id == d, 'bid_price_per_MWh'].values[0] * pd[d]
        for d in demands.id
    )
    - gp.quicksum(
        generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0] * pg[g]
        for g in generators.id
    ),
    GRB.MAXIMIZE
)

# -------------------------------------------------------------------
# 6. Power balance constraint (copper-plate)
# -------------------------------------------------------------------

power_balance = model.addConstr(
    gp.quicksum(pd[d] for d in demands.id)
    - gp.quicksum(pg[g] for g in generators.id)
    - gp.quicksum(pw[w] for w in wind_farms.id) ==  0,
    name='power_balance'
)

# Explicit production range constraints for generators and wind farms
for g in generators.id:
    g_cap = generators.loc[generators.id == g, 'capacity_MW'].values[0]
    model.addConstr(pg[g] >= 0, name=f'pg_min_{g}')
    model.addConstr(pg[g] <= g_cap, name=f'pg_max_{g}')

for w in wind_farms.id:
    w_cap = wind_farms.loc[wind_farms.id == w, 'capacity_MW'].values[0]
    model.addConstr(pw[w] >= 0, name=f'pw_min_{w}')
    model.addConstr(pw[w] <= w_cap, name=f'pw_max_{w}')

for d in demands.id:
    d_cons = demands.loc[demands.id == d, 'consumption_MW'].values[0]
    model.addConstr(pd[d] >= 0, name=f'pd_min_{d}')
    model.addConstr(pd[d] <= d_cons, name=f'pd_max_{d}')


# -------------------------------------------------------------------
# 7. Solve the optimization problem
# -------------------------------------------------------------------

model.optimize()

# -------------------------------------------------------------------
# 8. Extract market results
# -------------------------------------------------------------------

market_price = power_balance.Pi
clearing_quantity = sum(pg[g].X for g in generators.id) + sum(pw[w].X for w in wind_farms.id)

print('\n================ MARKET RESULTS ================')
print(f'Market-clearing price: {market_price:.2f} EUR/MWh\n')


# -------------------------------------------------------------------
# 9. Plot curves and solved clearing point
# -------------------------------------------------------------------

plt.figure(figsize=(12, 8))

plt.step(
    supply['cum_quantity'],
    supply['prod_cost_per_MWh'],
    where='post',
    color='red',
    label='Supply'
)

plt.step(
    demands_sorted['cum_quantity'],
    demands_sorted['bid_price_per_MWh'],
    where='post',
    color='black',
    label='Demand'
)
plt.xlabel('Quantity (MW)', fontsize = 15)
plt.ylabel('Price (EUR/MWh)', fontsize = 15)
plt.title('Merit-Order Supply and Demand Curves')
plt.legend(fontsize = 15)
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# 10. Generator dispatch and profits
# -------------------------------------------------------------------

print('Conventional generators:')
total_generation_cost = 0.0

for g in generators.id:
    output = pg[g].X
    cost = generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0]
    profit = (market_price - cost) * output
    total_generation_cost += cost * output

    print(
        f' Gen {g:2d} | Output: {output:7.2f} MW '
        f'| Cost: {cost:6.2f} EUR/MWh '
        f'| Profit: {profit:9.2f} EUR'
    )

# -------------------------------------------------------------------
# 11. Wind farm dispatch and profits
# -------------------------------------------------------------------

print('\nWind farms:')

for w in wind_farms.id:
    output = pw[w].X
    profit = market_price * output  # zero marginal cost

    print(
        f' Wind {w:2d} | Output: {output:7.2f} MW '
        f'| Profit: {profit:9.2f} EUR'
    )

# -------------------------------------------------------------------
# 12. Demand utility
# -------------------------------------------------------------------

print('\nDemands:')

total_utility = 0.0

for d in demands.id:
    served = pd[d].X
    bid = demands.loc[demands.id == d, 'bid_price_per_MWh'].values[0]
    utility = served * (bid - market_price)
    total_utility += utility

    print(
        f' Load {d:2d} | Served: {served:7.2f} MW '
        f'| Utility: {utility:9.2f} EUR'
    )

# -------------------------------------------------------------------
# 13. System-level metrics
# -------------------------------------------------------------------

social_welfare = model.objVal

print('\n================ SYSTEM METRICS ================')
print(f'Total generation cost: {total_generation_cost:.2f} EUR')
print(f'Total demand utility:  {total_utility:.2f} EUR')
print(f'Social welfare:        {social_welfare:.2f} EUR')
print('================================================\n')

# -------------------------------------------------------------------
# 14. KKT verification
# -------------------------------------------------------------------

print("\n================ KKT VERIFICATION (FULL) ================\n")

tolerance = 1e-5
lambda_dual = power_balance.Pi
market_price_kkt = -lambda_dual

print(f"Lambda (dual of balance): {lambda_dual:.6f}")
print(f"Market price from dual:  {market_price_kkt:.6f}\n")

# -----------------------
# Marginal generator
# -----------------------

for g in generators.id:
    output = pg[g].X
    cap = generators.loc[generators.id == g, 'capacity_MW'].values[0]
    cost = generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0]

    if output > tolerance and output < cap - tolerance:
        print(f"Marginal generator found: {g}")
        print(f"Cost c_g = {cost:.6f}")
        print(f"Stationarity implies lambda = -c_g = {-cost:.6f}")
        print()

# -----------------------
# Marginal demand
# -----------------------

for d in demands.id:
    served = pd[d].X
    max_d = demands.loc[demands.id == d, 'consumption_MW'].values[0]
    bid = demands.loc[demands.id == d, 'bid_price_per_MWh'].values[0]

    if served > tolerance and served < max_d - tolerance:
        print(f"Marginal demand found: {d}")
        print(f"Bid b_d = {bid:.6f}")
        print(f"Stationarity implies lambda = -b_d = {-bid:.6f}")

print("=========================================================\n") 

def calculate_balancing_market(
    generators,
    wind_farms,
    demands,
    pg,
    pw,
    pd,
    market_price,
):
    """Compute balancing market clearing price.

    Parameters
    ----------
    generators, wind_farms, demands, supply : pandas.DataFrame
        Input datasets used to configure the optimization.
    pg, pw, pd : Gurobi variables
        Day‑ahead dispatch results (solutions already available via .X).
    market_price : float
        Day‑ahead market‑clearing price (EUR/MWh) used by the two‑price
        scheme when appropriate.
    strategy : bool, optional
        ``False`` for the classic single‑price balancing market; ``True``
        to activate the two‑price scheme. In the latter case the price is
        replaced by ``market_price`` whenever one of the following two
        conditions occurs:

          * there is a power deficit (imbalance > 0) and actual generation
            exceeds the day‑ahead forecast; or
          * there is a power surplus (imbalance < 0) and actual generation
            is below the day‑ahead forecast.

    Returns
    -------
    float
        Balancing market clearing price (EUR/MWh).
    """
    # --- Wind deviations ---
    wind_ids = list(wind_farms.id)
    wind_realized = {}
    for i, w in enumerate(wind_ids):
        if i < 2:
            wind_realized[w] = 0.85 * pw[w].X   # -15%
        else:
            wind_realized[w] = 1.10 * pw[w].X   # +10%

    # --- Generator outage ---
    gen_ids = list(generators.id)
    outage_gen = gen_ids[7]  # outage in first generator
    generators.loc[generators.id == outage_gen, 'capacity_MW'] = 0.0  # set capacity to zero
    pg_realized = {}
    for g in gen_ids:
        if g == outage_gen:
            pg_realized[g] = 0.0
        else:
            pg_realized[g] = pg[g].X

    # Compute total imbalance
    total_generation_realized = sum(pg_realized[g] for g in gen_ids) + \
                                 sum(wind_realized[w] for w in wind_ids)
    total_demand_DA = sum(pd[d].X for d in demands.id)
    imbalance = total_demand_DA - total_generation_realized

    print("\n===============================")
    print("Intraday market clearance")
    print("===============================")
    print(f"Generator in outage: {outage_gen}")
    print(f"System imbalance (MW): {imbalance:.2f}")

    # ============================================================
    # BALANCING MARKET OPTIMIZATION
    # ============================================================
    bal_model = gp.Model("BalancingMarket")

    # -------------------------
    # Variables
    # -------------------------
    # Upward regulation generators
    p_up = bal_model.addVars(generators.id, lb=0, name="production_upward")

    # Downward regulation generators
    p_down = bal_model.addVars(generators.id, lb=0,
                               ub=generators.set_index('id').capacity_MW,
                               name="production_downward")

    # Load curtailment
    p_curt = bal_model.addVars(demands.id, lb=0, name="demand_curtailment")


    # Upward and downward regulation prices
    up_cost = {}
    down_cost = {}
    
    for g in generators.id:
        # (Marginal Cost)
        mc = generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0]
        
        # max(λ_DA + 0.1 * MC, MC)
        up_cost[g] = max(market_price + 0.10 * mc, mc)
        
        # downward regulation price remains the same as balancing price
        down_cost[g] = market_price - 0.15 * mc
    # -------------------------
    # Objective (MINIMIZE)
    # -------------------------
    bal_model.setObjective(
        gp.quicksum(up_cost[g] * p_up[g] for g in generators.id)
        + gp.quicksum(
            demands.loc[demands.id == d, 'curtailment_cost_per_MWh'].iloc[0] * p_curt[d]
            for d in demands.id
        )
        - gp.quicksum(down_cost[g] * p_down[g] for g in generators.id),
        GRB.MINIMIZE,
    )

    # -------------------------
    # Balancing constraint
    # -------------------------
    bal_constraint = bal_model.addConstr(
        gp.quicksum(p_up[g] - p_down[g] for g in generators.id)
        + gp.quicksum(p_curt[d] for d in demands.id)
        == imbalance,
        name="balancing_equation",
    )

    # -------------------------
    # Capacity limits
    # -------------------------
    for g in generators.id:
        cap = generators.loc[generators.id == g, 'capacity_MW'].values[0]
        # Up limited by remaining headroom
        bal_model.addConstr(pg_realized[g] + p_up[g] <= cap)
        # Down limited by actual production
        bal_model.addConstr(p_down[g] <= pg_realized[g])
    for d in demands.id:
        bal_model.addConstr(p_curt[d] <= pd[d].X)

    bal_model.setParam('OutputFlag', 0)
    bal_model.optimize()

    balancing_price = -bal_constraint.Pi

    # -------------------------------------------------------------------
    # Calculate profits under one-price and two-price schemes
    # -------------------------------------------------------------------
    balancing_price = abs(bal_constraint.Pi)
    print(f"\n================ Market Prices ================")
    print(f"DA Price: {market_price:.2f} €/MWh")
    print(f"Balancing Price: {balancing_price:.2f} €/MWh")

    if bal_model.status == GRB.OPTIMAL:
        tso_total_cost = bal_model.ObjVal
        print(f"\nTotal Balancing Cost: {tso_total_cost:.2f} €")

        up_total = sum(up_cost[g] * p_up[g].X for g in generators.id)
        dn_total = sum(down_cost[g] * p_down[g].X for g in generators.id)
        curt_total = sum(500.0 * p_curt[d].X for d in demands.id)
        print(f" - upward regulation cost: {up_total:.2f} €")
        print(f" - downward regulation cost: {dn_total:.2f} €")
        print(f" - load curtailment cost: {curt_total:.2f} €")
        print(f" - validation objective function value (Up - Dn + Curt): {up_total - dn_total + curt_total:.2f} €")
    # judge the system state: imbalance > 0 (Short)，imbalance < 0 (Long)
    system_short = imbalance > 0
    system_long = imbalance < 0

    # -------------------------------------------------------------------
    # define a function to calculate profits under both single-price and two-price schemes
    # -------------------------------------------------------------------
    def calculate_profits(scheme="one-price"):
        profits = {"generators": {}, "wind_farms": {}}
        
        # 1. calculate generator profits
        for g in generators.id:
            # day-ahead revenue
            da_revenue = market_price * pg[g].X
            
            # generator's actual output considering outage and balancing actions
            if g == outage_gen:
                actual_gen = 0.0  # failed generator output is 0
            else:
                actual_gen = pg[g].X + p_up[g].X - p_down[g].X
                
            cost = generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0] * actual_gen
            
            # balancing market clearance revenue/penalty
            if g == outage_gen:
                # failed generator: only has imbalance deviation (no balancing service provided)
                deviation = actual_gen - pg[g].X  # negative deviation -> produced less than DA schedule
                
                if scheme == "one-price":
                    # single-price mechanism: all deviations settled at balancing price
                    imb_revenue = deviation * balancing_price
                elif scheme == "two-price":
                    # two-price mechanism: judge whether it exacerbates or alleviates system imbalance
                    if system_short and deviation < 0:
                        imb_revenue = deviation * balancing_price # exacerbate shortage, heavy penalty (buy back unsupplied power at expensive balancing price)
                    elif system_long and deviation < 0:
                        imb_revenue = deviation * market_price    # alleviate surplus, light penalty (buy back at day-ahead price)
                    else:
                        imb_revenue = deviation * balancing_price # fallback logic
            else:
                # flexible generator: dispatch deviation (provides balancing service)
                # strictly follow dispatch instructions, directly use balancing price for rewards/punishments
                imb_revenue = p_up[g].X * balancing_price - p_down[g].X * balancing_price
                
            profits["generators"][g] = da_revenue - cost + imb_revenue

        # 2. calculate wind farm profits
        for w in wind_farms.id:
            da_revenue = market_price * pw[w].X
            cost = 0.0 # wind has zero marginal cost
            
            # inbalance deviation for wind farm
            deviation = wind_realized[w] - pw[w].X
            
            if scheme == "one-price":
                imb_revenue = deviation * balancing_price
            elif scheme == "two-price":
                if system_short:
                    if deviation < 0:
                        imb_revenue = deviation * balancing_price # less production, heavy penalty
                    else:
                        imb_revenue = deviation * market_price    # more production, light reward
                elif system_long:
                    if deviation > 0:
                        imb_revenue = deviation * balancing_price # more production, exacerbate surplus, penalty (balancing price is usually low at this time)
                    else:
                        imb_revenue = deviation * market_price    # less production, alleviate surplus, settle at day-ahead price
                else:
                    imb_revenue = deviation * balancing_price
                    
            profits["wind_farms"][w] = da_revenue - cost + imb_revenue
            
        return profits

    # -------------------------------------------------------------------
    # one-price vs two-price profit comparison
    # -------------------------------------------------------------------
    profits_one = calculate_profits(scheme="one-price")
    profits_two = calculate_profits(scheme="two-price")

    print("\n================ One-Price vs Two-Price ================")
    print(f"{'participant':<15} | {'One-Price Profit (€)':<20} | {'Two-Price Profit (€)':<20} | {'Difference (€)':<10}")
    print("-" * 75)
    
    # print generator comparison
    for g in generators.id:
        p1 = profits_one['generators'][g]
        p2 = profits_two['generators'][g]
        diff = p2 - p1
        
        # mark the generator with outage and those providing balancing service
        label = f"Gen {g}"
        if g == outage_gen:
            label += " (Outage)"
        elif p_up[g].X > 0 or p_down[g].X > 0:
            label += " (Bal Provider)"
            
        print(f"{label:<15} | {p1:<20.2f} | {p2:<20.2f} | {diff:<10.2f}")

    # print wind farm comparison
    for w in wind_farms.id:
        p1 = profits_one['wind_farms'][w]
        p2 = profits_two['wind_farms'][w]
        diff = p2 - p1
        
        # mark wind farms with positive or negative deviation
        dev_dir = " (Short)" if wind_realized[w] < pw[w].X else " (Long)"
        label = f"Wind {w}" + dev_dir
        
        print(f"{label:<15} | {p1:<20.2f} | {p2:<20.2f} | {diff:<10.2f}")

    # -------------------------------------------------------------------
    # ONLY Balancing Market: one-price vs two-price profit comparison
    # -------------------------------------------------------------------
    def calculate_balancing_profits(scheme="one-price"):
        bal_profits = {"generators": {}, "wind_farms": {}}
        
        # 1. calculate generator balancing market profits / deviation penalties
        for g in generators.id:
            if g == outage_gen:
                actual_gen = 0.0
                deviation = actual_gen - pg[g].X
                
                if scheme == "one-price":
                    imb_revenue = deviation * balancing_price
                elif scheme == "two-price":
                    if system_short and deviation < 0:
                        imb_revenue = deviation * balancing_price
                    elif system_long and deviation < 0:
                        imb_revenue = deviation * market_price
                    else:
                        imb_revenue = deviation * balancing_price
            else:
                actual_mc = generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0]
                # balance provider：
                # TSO pays for upward regulation, charges for downward regulation, net cash flow is:
                cash_flow = p_up[g].X * balancing_price - p_down[g].X * balancing_price
                
                # cost change due to deviation from DA schedule
                cost_change = p_up[g].X * actual_mc - p_down[g].X * actual_mc
                
                # balancing market net profit
                imb_revenue = cash_flow - cost_change
                
            bal_profits["generators"][g] = imb_revenue

        # 2. calculate wind farm balancing market profits / deviation penalties
        for w in wind_farms.id:
            deviation = wind_realized[w] - pw[w].X
            
            if scheme == "one-price":
                imb_revenue = deviation * balancing_price
            elif scheme == "two-price":
                if system_short:
                    if deviation < 0:
                        imb_revenue = deviation * balancing_price
                    else:
                        imb_revenue = deviation * market_price
                elif system_long:
                    if deviation > 0:
                        imb_revenue = deviation * balancing_price
                    else:
                        imb_revenue = deviation * market_price
                else:
                    imb_revenue = deviation * balancing_price
                    
            bal_profits["wind_farms"][w] = imb_revenue
            
        return bal_profits

    bal_profits_one = calculate_balancing_profits(scheme="one-price")
    bal_profits_two = calculate_balancing_profits(scheme="two-price")

    print("\n================ Balancing Market Profits Only (One-Price vs Two-Price) ================")
    print(f"{'Participant':<18} | {'One-Price Bal Profit (€)':<25} | {'Two-Price Bal Profit (€)':<25} | {'Difference (€)':<15}")
    print("-" * 90)
    
    # print generator comparison
    for g in generators.id:
        p1 = bal_profits_one['generators'][g]
        p2 = bal_profits_two['generators'][g]
        diff = p2 - p1
        
        label = f"Gen {g}"
        if g == outage_gen:
            label += " (Outage)"
        elif p_up[g].X > 0 or p_down[g].X > 0:
            label += " (Bal Provider)"
            
        print(f"{label:<18} | {p1:<25.2f} | {p2:<25.2f} | {diff:<15.2f}")

    # print wind farm comparison
    for w in wind_farms.id:
        p1 = bal_profits_one['wind_farms'][w]
        p2 = bal_profits_two['wind_farms'][w]
        diff = p2 - p1
        
        dev_dir = " (Short)" if wind_realized[w] < pw[w].X else " (Long)"
        label = f"Wind {w}" + dev_dir
        
        print(f"{label:<18} | {p1:<25.2f} | {p2:<25.2f} | {diff:<15.2f}")
    

    return balancing_price


if __name__ == "__main__":
    # check if all required variables are defined 
    try:
        print("\n" + "="*50)
        print(">>> executing balancing market analysis (Step 5) ...")
        print("="*50)
        
        calculate_balancing_market(generators, wind_farms, demands, pg, pw, pd, market_price)
        
        print("\n>>> analysis completed successfully!")
        
    except NameError as e:
        print(f"\n[wrong]: calculation failed. Error message: {e}")







