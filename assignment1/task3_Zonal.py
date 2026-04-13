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

# Define zones
zone1 = [1,2,3,4,5,9,11,14,15,16,17,18,19,21,24]  # Arbitrarily picked
#zone2 = [b for b in buses if b not in zone1]
zone2 = [6,7,8,10,12,13,20,22,23]
zones = {1: zone1, 2: zone2}

# Map busses to zones
bus_to_zone = {b: 1 for b in zone1}
bus_to_zone.update({b: 2 for b in zone2})

# Find ATC
interzonal_lines = lines[
    (lines.from_node.isin(zone1) & lines.to_node.isin(zone2)) | 
    (lines.from_node.isin(zone2) & lines.to_node.isin(zone1))
]
ATC = interzonal_lines.capacity.sum()
# ===============================================================
# 2. Build optimization model
# ===============================================================

model = gp.Model('MarketClearing_Zonal')

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

flow12 = model.addVar(
    lb=-ATC,            # Flow from zone 1 to 2
    ub=ATC,
    name='flow_1_2'
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
# 6. Zonal power balance constraints
# ===============================================================

for n in zones:
    model.addConstr(
        gp.quicksum(
            pg[g] for g in generators.id
            if generators.loc[generators.id == g, 'location_node'].values[0] in zones[n]
        )
        + gp.quicksum(
            pw[w] for w in wind_farms.id
            if wind_farms.loc[wind_farms.id == w, 'location_node'].values[0] in zones[n]
        )
        - gp.quicksum(
            pd[d] for d in demands.id
            if demands.loc[demands.id == d, 'location_node'].values[0] in zones[n]
        )
        + (-flow12 if n==1 else flow12)

        == 0,
        name=f'zonal_balance_{n}'
    )

# ===============================================================
# 8. Solve
# ===============================================================

model.optimize()

# ===============================================================
# 9. Extract zonal prices (dual variables)
# ===============================================================

Zonal_price = {
    n: -model.getConstrByName(f'zonal_balance_{n}').Pi
    for n in zones
}

print('\n============= ZONAL PRICES =============')
for n in zones:
    print(f'Zone {n:2d}: {Zonal_price[n]:8.2f} EUR/MWh')
print('=======================================\n')

# ===============================================================
# 10. Profits & utilities
# ===============================================================

print('Conventional generators:')
for g in generators.id:
    bus = generators.loc[generators.id == g, 'location_node'].values[0]
    z = bus_to_zone[bus]
    output = pg[g].X
    cost = generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0]

    profit = (Zonal_price[z] - cost) * output

    print(
        f' Gen {g:2d} @ bus {bus:2d} (Zone {z})| '
        f'Output {output:7.2f} MW | '
        f'Profit {profit:9.2f} EUR'
    )

print('\nWind farms:')
for w in wind_farms.id:
    bus = wind_farms.loc[wind_farms.id == w, 'location_node'].values[0]
    z = bus_to_zone[bus]
    output = pw[w].X
    profit = Zonal_price[z] * output

    print(
        f' Wind {w:2d} @ bus {bus:2d} (Zone {z})| '
        f'Output {output:7.2f} MW | '
        f'Profit {profit:9.2f} EUR'
    )

print('\nDemands:')
for d in demands.id:
    bus = demands.loc[demands.id == d, 'location_node'].values[0]
    z = bus_to_zone[bus]
    served = pd[d].X
    bid = demands.loc[demands.id == d, 'bid_price_per_MWh'].values[0]
    utility = served * (bid - Zonal_price[z])

    print(
        f' Load {d:2d} @ bus {bus:2d} (Zone {z})| '
        f'Served {served:7.2f} MW | '
        f'Utility {utility:9.2f} EUR'
    )

print('\nSocial welfare:', model.objVal)
