import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os

# Import full-sample bids and profits from Tasks 1.1 and 1.2 as risk-neutral baselines
from task1_1 import run_task_1_1
from task1_2 import run_task_1_2

_, profit_neutral_1p = run_task_1_1()
_, profit_neutral_2p = run_task_1_2()

# ==========================================
# Task 1.4: Risk-Averse Offering Strategy
# CVaR with alpha = 0.90, beta swept from 0 upward
# Done for both one-price and two-price schemes
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'final_1600_scenarios_input.csv'
df_path = os.path.abspath(
    os.path.join(script_dir, '..', 'scenario_prep', file_name)
)

df = pd.read_csv(df_path)

hours        = sorted(df['Hour'].unique())
scenarios    = df['Scenario_ID'].unique()
P_max        = 500
ALPHA        = 0.90   # CVaR confidence level (fixed throughout)

# Raw (non-normalized) probabilities
prob     = df[['Scenario_ID', 'Probability']].drop_duplicates().set_index('Scenario_ID')['Probability'].to_dict()
DA_price = df.set_index(['Hour', 'Scenario_ID'])['DA_Price'].to_dict()
Bal_price= df.set_index(['Hour', 'Scenario_ID'])['Bal_Price'].to_dict()
Wind_MW  = df.set_index(['Hour', 'Scenario_ID'])['Wind_MW'].to_dict()

# Beta values to sweep: 0 (risk-neutral) → 1 (fully risk-averse)
betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ==========================================
# Helper: scenario profit under one-price scheme
# ==========================================
def scenario_profit_one_price(bids, t, w):
    return DA_price[t, w] * bids[t] + Bal_price[t, w] * (Wind_MW[t, w] - bids[t])


# ==========================================
# One-Price CVaR Model
# ==========================================
def solve_one_price_cvar(beta):
    """
    Maximise: (1 - beta) * E[profit] + beta * CVaR_alpha[profit]

    CVaR linearisation (profit maximisation → we want CVaR of profit, i.e.
    expected value in the worst (1-alpha) tail):

        CVaR = eta - 1/(1-alpha) * sum_w prob[w] * max(0, eta - profit_w)

    Auxiliary variable lambda_w >= 0 captures shortfall below VaR (eta):
        lambda_w >= eta - profit_w
        lambda_w >= 0

    CVaR = eta - 1/(1-alpha) * sum_w prob[w] * lambda_w
    """
    m = gp.Model("OnePriceCVaR")
    m.setParam('OutputFlag', 0)

    P_DA     = m.addVars(hours, lb=0, ub=P_max, name="P_DA")
    eta      = m.addVar(lb=0, name="eta")          # VaR estimate
    lambda_w = m.addVars(scenarios, lb=0, name="lambda")       # shortfall vars

    # Scenario profit expressions (linear in P_DA)
    profit_w = {
        w: gp.quicksum(
            DA_price[t, w] * P_DA[t] + Bal_price[t, w] * (Wind_MW[t, w] - P_DA[t])
            for t in hours
        )
        for w in scenarios
    }

    # Shortfall constraints: lambda_w >= eta - profit_w
    m.addConstrs(
        (lambda_w[w] >= eta - profit_w[w] for w in scenarios),
        name="shortfall"
    )

    # Expected profit term
    exp_profit = gp.quicksum(prob[w] * profit_w[w] for w in scenarios)

    # CVaR term
    cvar = eta - (1.0 / (1.0 - ALPHA)) * gp.quicksum(prob[w] * lambda_w[w] for w in scenarios)

    m.setObjective((1 - beta) * exp_profit + beta * cvar, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"One-price CVaR model did not converge for beta={beta}")

    bids       = {t: P_DA[t].X for t in hours}
    exp_val    = sum(prob[w] * sum(
                     DA_price[t, w] * bids[t] + Bal_price[t, w] * (Wind_MW[t, w] - bids[t])
                     for t in hours) for w in scenarios)
    cvar_val   = eta.X - (1.0 / (1.0 - ALPHA)) * sum(prob[w] * lambda_w[w].X for w in scenarios)

    return bids, exp_val, cvar_val


# ==========================================
# Two-Price CVaR Model
# ==========================================
def solve_two_price_cvar(beta):
    m = gp.Model("TwoPriceCVaR")
    m.setParam('OutputFlag', 0)

    P_DA        = m.addVars(hours, lb=0, ub=P_max, name="P_DA")
    delta_plus  = m.addVars(hours, scenarios, lb=0, name="delta_plus")
    delta_minus = m.addVars(hours, scenarios, lb=0, name="delta_minus")
    eta         = m.addVar(lb=0, name="eta")
    lambda_w    = m.addVars(scenarios, lb=0, name="lambda")

    # Imbalance decomposition
    m.addConstrs(
        (Wind_MW[t, w] - P_DA[t] == delta_plus[t, w] - delta_minus[t, w]
         for t in hours for w in scenarios),
        name="Imbalance"
    )

    # Scenario profit expressions
    profit_w = {
        w: gp.quicksum(
            DA_price[t, w] * P_DA[t]
            + min(DA_price[t, w], Bal_price[t, w]) * delta_plus[t, w]
            - max(DA_price[t, w], Bal_price[t, w]) * delta_minus[t, w]
            for t in hours
        )
        for w in scenarios
    }

    # Shortfall constraints
    m.addConstrs(
        (lambda_w[w] >= eta - profit_w[w] for w in scenarios),
        name="shortfall"
    )

    exp_profit = gp.quicksum(prob[w] * profit_w[w] for w in scenarios)
    cvar       = eta - (1.0 / (1.0 - ALPHA)) * gp.quicksum(prob[w] * lambda_w[w] for w in scenarios)

    m.setObjective((1 - beta) * exp_profit + beta * cvar, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price CVaR model did not converge for beta={beta}")

    bids = {t: P_DA[t].X for t in hours}

    # Recompute exp profit and CVaR from solution values
    exp_val = sum(
        prob[w] * sum(
            DA_price[t, w] * bids[t]
            + min(DA_price[t, w], Bal_price[t, w]) * max(0, Wind_MW[t, w] - bids[t])
            - max(DA_price[t, w], Bal_price[t, w]) * max(0, bids[t] - Wind_MW[t, w])
            for t in hours
        )
        for w in scenarios
    )
    cvar_val = eta.X - (1.0 / (1.0 - ALPHA)) * sum(prob[w] * lambda_w[w].X for w in scenarios)

    return bids, exp_val, cvar_val


# ==========================================
# Sweep beta for both schemes
# ==========================================
results_1p = []
results_2p = []

print("=" * 60)
print(f"Task 1.4: Risk-Averse CVaR Sweep  (alpha = {ALPHA})")
print("=" * 60)
print(f"{'Beta':>6} | {'E[P] 1-price':>14} | {'CVaR 1-price':>14} | {'E[P] 2-price':>14} | {'CVaR 2-price':>14}")
print("-" * 70)

for beta in betas:
    bids_1p, exp_1p, cvar_1p = solve_one_price_cvar(beta)
    bids_2p, exp_2p, cvar_2p = solve_two_price_cvar(beta)

    results_1p.append({'beta': beta, 'exp_profit': exp_1p, 'cvar': cvar_1p, 'bids': bids_1p})
    results_2p.append({'beta': beta, 'exp_profit': exp_2p, 'cvar': cvar_2p, 'bids': bids_2p})

    print(f"{beta:>6.2f} | {exp_1p:>14,.2f} | {cvar_1p:>14,.2f} | {exp_2p:>14,.2f} | {cvar_2p:>14,.2f}")

print("-" * 70)


# ==========================================
# Plot: Expected Profit vs CVaR (efficient frontier)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, results, label, color, neutral_profit in [
    (axes[0], results_1p, "One-Price",  "steelblue", profit_neutral_1p),
    (axes[1], results_2p, "Two-Price",  "seagreen",  profit_neutral_2p),
]:
    exp_profits = [r['exp_profit'] for r in results]
    cvars       = [r['cvar']       for r in results]

    ax.plot(cvars, exp_profits, 'o-', color=color, linewidth=2, markersize=7)

    # Annotate beta values
    for r in results:
        ax.annotate(
            f"β={r['beta']:.1f}",
            xy=(r['cvar'], r['exp_profit']),
            xytext=(5, 3), textcoords='offset points', fontsize=7
        )

    # Risk-neutral baseline (beta=0)
    ax.axhline(neutral_profit, color='red', linestyle='--', linewidth=1.2,
               label=f"Risk-neutral baseline (1.1/1.2): {neutral_profit:,.0f} €")

    ax.set_title(f"Task 1.4 – {label} Scheme\nExpected Profit vs CVaR$_{{α={ALPHA}}}$")
    ax.set_xlabel("CVaR (€)  [higher = less risky]")
    ax.set_ylabel("Expected Profit (€)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)

plt.tight_layout()
#plt.savefig('task1_4_efficient_frontier.png', dpi=150)
plt.show()


# ==========================================
# Plot: Hourly bids for beta=0 vs beta=1 (both schemes)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, results, label, c0, c1 in [
    (axes[0], results_1p, "One-Price", "steelblue", "coral"),
    (axes[1], results_2p, "Two-Price", "seagreen",  "salmon"),
]:
    bids_b0 = results[0]['bids']   # beta = 0
    bids_b1 = results[-1]['bids']  # beta = 1

    ax.step(hours, [bids_b0[t] for t in hours], where='mid', color=c0,
            linewidth=2, label="β=0 (risk-neutral)")
    ax.step(hours, [bids_b1[t] for t in hours], where='mid', color=c1,
            linewidth=2, linestyle='--', label="β=1 (fully risk-averse)")

    ax.set_title(f"Task 1.4 – {label} Scheme\nHourly DA Bids: β=0 vs β=1")
    ax.set_xlabel("Hour")
    ax.set_ylabel("DA Bid (MW)")
    ax.set_xticks(hours)
    ax.set_ylim(-10, P_max + 20)
    ax.legend()
    ax.grid(alpha=0.4)

plt.tight_layout()
#plt.savefig('task1_4_bids_comparison.png', dpi=150)
plt.show()

print("Task 1.4 complete.")