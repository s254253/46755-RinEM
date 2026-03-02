import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================================================
# 1. Read data
# ===============================================================

generators = pd.read_csv('conventional_generators.txt', sep=',')
wind_farms = pd.read_csv('wind_farms.txt', sep=',')
demands = pd.read_csv('demands.txt', sep=',')
lines = pd.read_csv('transmission2.txt', sep=',')
print(lines.head())

wind_farms['prod_cost_per_MWh'] = 0.0
wind_farms.rename(columns={'day_ahead_forecast_MW': 'capacity_MW'}, inplace=True)

demands['bid_price_per_MWh'] = (
    (np.exp(-np.linspace(0, 5, len(demands))) - np.exp(-6))
    / (1 - np.exp(-6))
    * 237
)

# Network sets
buses = sorted(set(lines.from_node).union(set(lines.to_node)))
lines['id'] = lines.index

# ===============================================================
# 2. Build optimization model
# ===============================================================

model = gp.Model('MarketClearing_Network')

# ===============================================================
# 3. Decision variables
# ===============================================================

pg = model.addVars(
    generators.id,
    lb=0,
    ub=generators.set_index('id').capacity_MW,
    name='pg'
)

pw = model.addVars(
    wind_farms.id,
    lb=0,
    ub=wind_farms.set_index('id').capacity_MW,
    name='pw'
)

pd = model.addVars(
    demands.id,
    lb=0,
    ub=demands.set_index('id').consumption_MW,
    name='pd'
)

theta = model.addVars(
    buses,
    lb=-GRB.INFINITY, 
    name='theta'
)

flow = model.addVars(
    lines.id,
    lb=-GRB.INFINITY,
    name='flow'
)

# ===============================================================
# 4. Objective: maximize social welfare
# ===============================================================

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

# ===============================================================
# 5. DC power flow constraints
# ===============================================================

for _, row in lines.iterrows():
    l = row.id
    i = int(row.from_node)
    j = int(row.to_node)
    b = 1/row.susceptance

    model.addConstr(
        flow[l] == 100 * b * (theta[i] - theta[j]),
        name=f'dc_flow_{l}'
    )

    model.addConstr(flow[l] <= row.capacity, name=f'line_max_{l}')
    model.addConstr(flow[l] >= -row.capacity, name=f'line_min_{l}')

# ===============================================================
# 6. Nodal power balance constraints
# ===============================================================

for n in buses:
    model.addConstr(
        gp.quicksum(
            pg[g] for g in generators.id
            if generators.loc[generators.id == g, 'location_node'].values[0] == n
        )
        + gp.quicksum(
            pw[w] for w in wind_farms.id
            if wind_farms.loc[wind_farms.id == w, 'location_node'].values[0] == n
        )
        - gp.quicksum(
            pd[d] for d in demands.id
            if demands.loc[demands.id == d, 'location_node'].values[0] == n
        )
        + gp.quicksum(
            flow[l] for l in lines.id
            if lines.loc[l, 'to_node'] == n
        )
        - gp.quicksum(
            flow[l] for l in lines.id
            if lines.loc[l, 'from_node'] == n
        )
        == 0,
        name=f'nodal_balance_{n}'
    )

# ===============================================================
# 7. Slack bus
# ===============================================================

slack_bus = buses[12]
model.addConstr(theta[slack_bus] == 0, name='slack_bus')

# ===============================================================
# 8. Solve
# ===============================================================

model.optimize()

# ===============================================================
# 9. Extract nodal prices (dual variables)
# ===============================================================

lmp = {
    n: -model.getConstrByName(f'nodal_balance_{n}').Pi
    for n in buses
}

print('\n============= NODAL PRICES =============')
for n in buses:
    print(f'Bus {n:2d}: {lmp[n]:8.2f} EUR/MWh')
print('=======================================\n')

# ===============================================================
# Plot nodal prices
# ===============================================================

# create bar chart of nodal prices
buses_sorted = sorted(buses)
prices = [lmp[n] for n in buses_sorted]

plt.figure()
plt.bar(buses_sorted, prices, color='skyblue')
plt.xlabel('Bus number')
plt.ylabel('Nodal price (EUR/MWh)')
plt.title('Nodal Prices by Bus')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ===============================================================
# 10. Profits & utilities (same logic as before)
# ===============================================================

print('Conventional generators:')
for g in generators.id:
    bus = generators.loc[generators.id == g, 'location_node'].values[0]
    output = pg[g].X
    cost = generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0]
    profit = (lmp[bus] - cost) * output

    print(
        f' Gen {g:2d} @ bus {bus:2d} | '
        f'Output {output:7.2f} MW | '
        f'Profit {profit:9.2f} EUR'
    )

print('\nWind farms:')
for w in wind_farms.id:
    bus = wind_farms.loc[wind_farms.id == w, 'location_node'].values[0]
    output = pw[w].X
    profit = lmp[bus] * output

    print(
        f' Wind {w:2d} @ bus {bus:2d} | '
        f'Output {output:7.2f} MW | '
        f'Profit {profit:9.2f} EUR'
    )

print('\nDemands:')
for d in demands.id:
    bus = demands.loc[demands.id == d, 'location_node'].values[0]
    served = pd[d].X
    bid = demands.loc[demands.id == d, 'bid_price_per_MWh'].values[0]
    utility = served * (bid - lmp[bus])

    print(
        f' Load {d:2d} @ bus {bus:2d} | '
        f'Served {served:7.2f} MW | '
        f'Utility {utility:9.2f} EUR'
    )

print('\nSocial welfare:', model.objVal)
