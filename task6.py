import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Add reserve capacities and offer prices for regulation
R_plus_map = generators.set_index('id')['max_up_res_MW'].to_dict()
R_minus_map = generators.set_index('id')['max_down_res_MW'].to_dict()
C_plus_map = generators.set_index('id')['up_res_cost_per_MW'].to_dict()
C_minus_map = generators.set_index('id')['down_res_cost_per_MW'].to_dict()

generators["R_plus"] = generators["id"].map(R_plus_map)
generators["R_minus"] = generators["id"].map(R_minus_map)
generators["C_plus"] = generators["id"].map(C_plus_map)
generators["C_minus"] = generators["id"].map(C_minus_map)
# ============================================================
# RESERVE MARKET OPTIMIZATION
# ============================================================
reserve_model = gp.Model("ReserveMarket")

# -------------------------------------------------------------
# Variables
# ---------------------------------------------------------------

# Limits for the reserve capacity given in the problem formulation
R_plus = generators.set_index("id")["R_plus"].to_dict()
R_minus = generators.set_index("id")["R_minus"].to_dict()

r_up = reserve_model.addVars(generators.id, lb=0, ub=R_plus, name="r_up")
r_down = reserve_model.addVars(generators.id, lb=0, ub=R_minus, name="r_down")

# --------------------------------------------------------------
# Objective (MINIMIZE)
# --------------------------------------------------------------

# Offer prices for upward and downward regulation
C_plus = generators.set_index("id")["C_plus"].to_dict()
C_minus = generators.set_index("id")["C_minus"].to_dict()

reserve_model.setObjective(
        gp.quicksum(C_plus[g] * r_up[g] for g in generators.id)
        + gp.quicksum(C_minus[g] * r_down[g] for g in generators.id),
        GRB.MINIMIZE,
    )

# -------------------------------------------------------------
# Reserve requirements
# -------------------------------------------------------------
total_demand = demands["consumption_MW"].sum()
up_req = 0.15
down_req = 0.10

URS = up_req * total_demand     # Upward Reserve Services
DRS = down_req * total_demand   # Downward Reserve Services

up_res_req = reserve_model.addConstr(
    gp.quicksum(r_up[g] for g in generators.id) == URS,
    name="UpwardReserveRequirement"
)

down_res_req = reserve_model.addConstr(
    gp.quicksum(r_down[g] for g in generators.id) == DRS,
    name="DownwardReserveRequirement"
)

# Capacity constraints for reserves
P_max = generators.set_index("id")["capacity_MW"].to_dict()
for g in generators.id:
    reserve_model.addConstr(r_up[g] + r_down[g] <= P_max[g], name=f"ReserveCapacity_{g}")

reserve_model.optimize()

reserve_up_price_clearing = up_res_req.Pi
reserve_down_price_clearing = down_res_req.Pi

reserve_results = {
    "r_up": {g: r_up[g].X for g in generators.id},
    "r_down": {g: r_down[g].X for g in generators.id},
    "up_price": up_res_req.Pi,
    "down_price": down_res_req.Pi,
    "URS": URS,
    "DRS": DRS
}

reserve_model.optimize()

if reserve_model.status == GRB.OPTIMAL:
    reserve_up_price_clearing = up_res_req.Pi
    reserve_down_price_clearing = down_res_req.Pi
    reserve_market_sw = reserve_model.ObjVal
else:
    print("Reserve model did not solve to optimality.")

print('\n================ RESERVE MARKET RESULTS ================')
print(f"Upward reserve clearing price: {up_res_req.Pi:.2f} EUR/MW")
print(f"Downward reserve clearing price: {down_res_req.Pi:.2f} EUR/MW")
print(f"Social welfare for reserve market: {reserve_model.objVal:.2f} EUR")

print("\nReserve provision by generator:")
for g in generators.id:
    print(
        f"Gen {g:2d} | Up reserve: {r_up[g].X:7.2f} MW | Down reserve: {r_down[g].X:7.2f} MW"
    )

# ===================================================================
# Build day-ahead optimization model
# ===================================================================
model = gp.Model('CopperPlate_MarketClearing_Step6_DA')

# Reserve results from reserve market
r_up_sol = {g: r_up[g].X for g in generators.id}
r_down_sol = {g: r_down[g].X for g in generators.id}
P_max = generators.set_index("id")["capacity_MW"].to_dict()

# Day-ahead bounds for conventional generators:
# pg[g] >= reserved downward capacity
# pg[g] <= capacity - reserved upward capacity
pg_lb = {g: r_down_sol[g] for g in generators.id}
pg_ub = {g: P_max[g] - r_up_sol[g] for g in generators.id}

# pg[g]: power produced by conventional generator g [MW]
pg = model.addVars(
    generators.id,
    lb=pg_lb,
    ub=pg_ub,
    name='pg'
)

# pw[w]: power produced by wind farm w [MW]
pw = model.addVars(
    wind_farms.id,
    lb=0,
    ub=wind_farms.set_index('id').capacity_MW.to_dict(),
    name='pw'
)

# pd[d]: served demand d [MW]
pd = model.addVars(
    demands.id,
    lb=0,
    ub=demands.set_index('id').consumption_MW.to_dict(),
    name='pd'
)
# -------------------------------------------------------------------
# Objective function: maximize social welfare
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
# Power balance constraint (copper-plate)
# -------------------------------------------------------------------
power_balance = model.addConstr(
    gp.quicksum(pd[d] for d in demands.id)
    - gp.quicksum(pg[g] for g in generators.id)
    - gp.quicksum(pw[w] for w in wind_farms.id)
    == 0,
    name='power_balance'
)
# -------------------------------------------------------------------
# 7. Solve the optimization problem
# -------------------------------------------------------------------
model.optimize()

market_price = power_balance.Pi
day_ahead_sw = model.ObjVal
total_social_welfare = day_ahead_sw - reserve_market_sw
print('\n================ DAY-AHEAD RESULTS ================')
print(f"Day-ahead market clearing price: {market_price:.2f} EUR/MWh")
print(f"Social welfare for European day-ahead market: {total_social_welfare:.2f} EUR ")

for g in generators.id:
    output = pg[g].X
    print(
        f'Gen {g:2d} | Output: {output:7.2f} MW '
        )

for w in wind_farms.id:
    output = pw[w].X
    profit = market_price * output  # zero marginal cost

    print(
        f' Wind {w:2d} | Output: {output:7.2f} MW '
        f'| Profit: {profit:9.2f} EUR'
    )

total_utility = 0.0

for d in demands.id:
    served = pd[d].X
    utility = demands.loc[demands.id == d, 'bid_price_per_MWh'].values[0] * served
    total_utility += utility

    print(
        f' Load {d:2d} | Served: {served:7.2f} MW '
        f'| Utility: {utility:9.2f} EUR'
    )
# ============================================================
# RESERVE MARKET OPTIMIZATION (US EDITION)
# ============================================================

USReserve_model = gp.Model('ReserveMarketUS')

# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------

# The up and down reserve of the generators
r_up = USReserve_model.addVars(generators.id, lb=0, ub=R_plus, name="r_up")
r_down = USReserve_model.addVars(generators.id, lb=0, ub=R_minus, name="r_down")

# pg[g]: power produced by conventional generator g [MW]
pg = USReserve_model.addVars(
    generators.id,
    lb=0,
    ub=generators.set_index('id').capacity_MW,
    name='pg'
)

# pw[w]: power produced by wind farm w [MW]
pw = USReserve_model.addVars(
    wind_farms.id,
    lb=0,
    ub=wind_farms.set_index('id').capacity_MW,
    name='pw'
)

# pd[d]: served demand d [MW]
pd = USReserve_model.addVars(
    demands.id,
    lb=0,
    ub=demands.set_index('id').consumption_MW,
    name='pd'
)

# --------------------------------------------------------------
# Objective (MAXIMIZE)
# --------------------------------------------------------------

USReserve_model.setObjective(
    gp.quicksum(
        demands.loc[demands.id == d, 'bid_price_per_MWh'].values[0] * pd[d]
        for d in demands.id
    )
    - gp.quicksum(
        generators.loc[generators.id == g, 'prod_cost_per_MWh'].values[0] * pg[g]
        for g in generators.id
    )
    - (
        gp.quicksum(C_plus[g] * r_up[g] for g in generators.id)
        + gp.quicksum(C_minus[g] * r_down[g] for g in generators.id)
    ),
    GRB.MAXIMIZE
)

# -------------------------------------------------------------------
# Constraints
# -------------------------------------------------------------------

# The power balance constraint
power_balance = USReserve_model.addConstr(
    gp.quicksum(pd[d] for d in demands.id)
    - gp.quicksum(pg[g] for g in generators.id)
    - gp.quicksum(pw[w] for w in wind_farms.id)
    == 0,
    name='power_balance'
)

# The up- and downward reserve requirements
up_res_req = USReserve_model.addConstr(
    gp.quicksum(r_up[g] for g in generators.id) == URS,
    name="UpwardReserveRequirement"
)

down_res_req = USReserve_model.addConstr(
    gp.quicksum(r_down[g] for g in generators.id) == DRS,
    name="DownwardReserveRequirement"
)

# Constraints ensuring that a generator cannot exceed its capacity (reserve + generation <= P_max) or produce less than its downward reserve.

for g in generators.id:
    USReserve_model.addConstr(
        pg[g] + r_up[g] <= P_max[g],
        name=f"Upwards_capacity_{g}"
    )

    USReserve_model.addConstr(
        pg[g] >= r_down[g],
        name=f"Downwards_capacity_{g}"
    )

# -------------------------------------------------------------------
# Solve the optimization problem
# -------------------------------------------------------------------

USReserve_model.optimize()

if USReserve_model.status == GRB.OPTIMAL:
    US_reserve_up_price_clearing = -up_res_req.Pi
    US_reserve_down_price_clearing = -down_res_req.Pi
    US_market_price = power_balance.Pi

    print('\n================ US RESULTS ================')
    print(f"Upward reserve clearing price: {US_reserve_up_price_clearing:.2f} EUR/MW")
    print(f"Downward reserve clearing price: {US_reserve_down_price_clearing:.2f} EUR/MW")
    print(f"Day-ahead market clearing price: {US_market_price:.2f} EUR/MWh")
    print(f"Social welfare for American day-ahead market: {USReserve_model.objVal:.2f} EUR")

    print("\nGenerator schedules:")
    for g in generators.id:
            print(
                f"Gen {g:2d} | "
                f"pg = {pg[g].X:7.2f} MW | "
                f"r_up = {r_up[g].X:7.2f} MW | "
                f"r_down = {r_down[g].X:7.2f} MW"
            )

else:
    print("US Reserve model did not solve to optimality.")

