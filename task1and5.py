import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Read data

generators = pd.read_csv('conventional_generators.txt', sep=',')
wind_farms = pd.read_csv('wind_farms.txt', sep=',')
demands = pd.read_csv('demands.txt', sep=',')

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
    - gp.quicksum(pw[w] for w in wind_farms.id),
    GRB.EQUAL,
    0,
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
plt.xlabel('Quantity (MW)')
plt.ylabel('Price (EUR/MWh)')
plt.title('Merit-Order Supply and Demand Curves')
plt.legend()
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


# ===============================================================
# 15. Market clearing with balancing market (--STEP 5--)
# ===============================================================

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
                           ub = generators.set_index('id').capacity_MW,
                           name="production_downward")

# Load curtailment
p_curt = bal_model.addVars(demands.id, lb=0, name="demand_curtailment")

# Wind adjustments
p_wind = bal_model.addVars(wind_farms.id, lb=0,
                           ub = [wind_realized[w] for w in wind_farms.id],
                            name="wind_adjustment")

# Upward and downward regulation prices
up_cost = {
    g: 1.10 * generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0]
    for g in generators.id
}

down_cost = {
    g: 0.85 * generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0]
    for g in generators.id
}

# -------------------------
# Objective (MINIMIZE)
# -------------------------

bal_model.setObjective(
    gp.quicksum(up_cost[g] * p_up[g] for g in generators.id)
    + gp.quicksum(demands.loc[demands.id == d, 'curtailment_cost_per_MWh'].iloc[0] * p_curt[d] for d in demands.id)
    - gp.quicksum(down_cost[g] * p_down[g] for g in generators.id),
    GRB.MINIMIZE
)

# -------------------------
# Balancing constraint
# -------------------------

bal_constraint = bal_model.addConstr(
    gp.quicksum(p_up[g] - p_down[g] for g in generators.id)
    + gp.quicksum(p_curt[d] for d in demands.id)
    + gp.quicksum(p_wind[w] for w in wind_farms.id)
    == imbalance,
    name="balancing_equation"
)

# -------------------------
# Capacity limits
# -------------------------

for g in generators.id:

    cap = generators.loc[generators.id == g, 'capacity_MW'].values[0]

    # Up limited by remaining headroom
    bal_model.addConstr(
        pg_realized[g] + p_up[g] <= cap
    )

    # Down limited by actual production
    bal_model.addConstr(
        p_down[g] <= pg_realized[g]
    )
for w in wind_farms.id:
    bal_model.addConstr(
        p_wind[w] <= wind_realized[w]
    )

for d in demands.id:
    bal_model.addConstr(p_curt[d] <= pd[d].X)

bal_model.optimize()

balancing_price = -bal_constraint.Pi

print(f"\nBalancing price with the single-price strategy: {balancing_price:.2f} EUR/MWh")