import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Read data 

generators = pd.read_csv('conventional_generators.txt',
                                       sep = ',')
wind_farms = pd.read_csv('wind_farms.txt',
                         sep = ',')  
demands = pd.read_csv('demands.txt', sep = ',')

wind_farms['prod_cost_per_MWh'] = 0.0  # zero marginal cost for wind as stated in first assumption 
wind_farms.rename(columns={"day_ahead_forecast_MW": "capacity_MW"}, inplace=True)

# Demand bid prices (descending from high to low)
demands["bid_price_per_MWh"] = np.linspace(500, 0, len(demands))

# -------------------------------------------------------------------
# 2. Build SUPPLY curve (Merit Order)
# -------------------------------------------------------------------

# Combine conventional + wind
supply = pd.concat([
    generators[["capacity_MW", "prod_cost_per_MWh"]],
    wind_farms[["capacity_MW", "prod_cost_per_MWh"]]
])

# Sort by increasing price, then increasing capacity
supply = supply.sort_values(
    by=["prod_cost_per_MWh", "capacity_MW"],
    ascending=[True, True]
).reset_index(drop=True)

# Create cumulative quantity
supply["cum_quantity"] = supply["capacity_MW"].cumsum()

# -------------------------------------------------------------------
# 3. Build DEMAND curve
# -------------------------------------------------------------------

# Sort by decreasing bid price
demands_sorted = demands.sort_values(
    by="bid_price_per_MWh",
    ascending=False
).reset_index(drop=True)

# Cumulative quantity
demands_sorted["cum_quantity"] = demands_sorted["consumption_MW"].cumsum()

# -------------------------------------------------------------------
# 4. Plot curves as step functions
# -------------------------------------------------------------------

plt.figure(figsize=(12, 8))

# Supply (increasing step curve)
plt.step(
    supply["cum_quantity"],
    supply["prod_cost_per_MWh"],
    where="post",
    color="red",
    label="Supply"
)

# Demand (decreasing step curve)
plt.step(
    demands_sorted["cum_quantity"],
    demands_sorted["bid_price_per_MWh"],
    where="post",
    color="black",
    label="Demand"
)

plt.xlabel("Quantity (MW)")
plt.ylabel("Price (€/MWh)")
plt.title("Merit-Order Supply and Demand Curves")
plt.legend()
plt.grid(True)

plt.show()

# -------------------------------------------------------------------
# 5. Decision variables
# -------------------------------------------------------------------
# pg[g]: power produced by conventional generator g [MW]
model = gp.Model("CopperPlate_MarketClearing_Step1")

pg = model.addVars(
    generators.id,
    lb=0, # lower bound is zero (no negative generation) as stated in assumption in announcement
    ub=generators.set_index("id").capacity_MW,
    name="pg"
)

# pw[w]: power produced by wind farm w [MW]
# Wind bids zero price and is limited by its forecast
pw = model.addVars(
    wind_farms.id,
    lb=0,
    ub=wind_farms.set_index("id").capacity_MW,
    name="pw"
)

# pd[d]: served demand d [MW]
# Curtailment is allowed but expensive (high bid price)
pd = model.addVars(
    demands.id,
    lb=0,
    ub=demands.set_index("id").consumption_MW,
    name="pd"
)

# -------------------------------------------------------------------
# 6. Objective function: maximize social welfare
# -------------------------------------------------------------------
# Social welfare = utility of demand – production cost of generators
# Wind generation has zero marginal cost

model.setObjective(
    gp.quicksum(
        demands.loc[demands.id == d, "bid_price_per_MWh"].values[0] * pd[d]
        for d in demands.id
    )
    -
    gp.quicksum(
        generators.loc[generators.id == g, "prod_cost_per_MWh"].values[0] * pg[g]
        for g in generators.id
    ),
    GRB.MAXIMIZE
)

# -------------------------------------------------------------------
# 7. Power balance constraint (copper-plate)
# -------------------------------------------------------------------
# Total generation = total consumption
# Dual variable of this constraint is the market-clearing price

power_balance = model.addConstr(
    gp.quicksum(pg[g] for g in generators.id)
    + gp.quicksum(pw[w] for w in wind_farms.id)
    ==
    gp.quicksum(pd[d] for d in demands.id),
    name="power_balance"
)

# -------------------------------------------------------------------
# 8. Solve the optimization problem
# -------------------------------------------------------------------
model.optimize()

# -------------------------------------------------------------------
# 9. Extract market results
# -------------------------------------------------------------------
# Market-clearing price (uniform price)
market_price = power_balance.Pi

print("\n================ MARKET RESULTS ================")
print(f"Market-clearing price: {market_price:.2f} €/MWh\n")

# -------------------------------------------------------------------
# 10. Generator dispatch and profits
# -------------------------------------------------------------------
print("Conventional generators:")
total_generation_cost = 0.0

for g in generators.id:
    output = pg[g].X
    cost = generators.loc[generators.id == g, "prod_cost_per_MWh"].values[0]
    profit = (market_price - cost) * output
    total_generation_cost += cost * output

    print(
        f" Gen {g:2d} | Output: {output:7.2f} MW "
        f"| Cost: {cost:6.2f} €/MWh "
        f"| Profit: {profit:9.2f} €"
    )

# -------------------------------------------------------------------
# 11. Wind farm dispatch and profits
# -------------------------------------------------------------------
print("\nWind farms:")

for w in wind_farms.id:
    output = pw[w].X
    profit = market_price * output  # zero marginal cost

    print(
        f" Wind {w:2d} | Output: {output:7.2f} MW "
        f"| Profit: {profit:9.2f} €"
    )

# -------------------------------------------------------------------
# 12. Demand utility
# -------------------------------------------------------------------
print("\nDemands:")

total_utility = 0.0

for d in demands.id:
    served = pd[d].X
    bid = demands.loc[demands.id == d, "curtailment_cost_per_MWh"].values[0]
    utility = served * (bid - market_price)
    total_utility += utility

    print(
        f" Load {d:2d} | Served: {served:7.2f} MW "
        f"| Utility: {utility:9.2f} €"
    )

# -------------------------------------------------------------------
# 13. System-level metrics
# -------------------------------------------------------------------
social_welfare = model.objVal

print("\n================ SYSTEM METRICS ================")
print(f"Total generation cost: {total_generation_cost:.2f} €")
print(f"Total demand utility:  {total_utility:.2f} €")
print(f"Social welfare:        {social_welfare:.2f} €")
print("================================================\n")
