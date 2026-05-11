import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os
import time

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

df_full = pd.read_csv(df_path)
df_full['DA_Price']  = df_full['DA_Price'].clip(lower=0)
df_full['Bal_Price'] = df_full['Bal_Price'].clip(lower=0)

# ==========================================
# Build full-sample parameter dicts (out-of-sample evaluation only)
# ==========================================
full_scenarios = df_full['Scenario_ID'].unique()
full_prob_raw  = df_full[['Scenario_ID', 'Probability']].drop_duplicates().set_index('Scenario_ID')['Probability']
full_prob      = full_prob_raw.to_dict()
full_DA        = df_full.set_index(['Hour', 'Scenario_ID'])['DA_Price'].to_dict()
full_bal       = df_full.set_index(['Hour', 'Scenario_ID'])['Bal_Price'].to_dict()
full_wind      = df_full.set_index(['Hour', 'Scenario_ID'])['Wind_MW'].to_dict()

hours = sorted(df_full['Hour'].unique())
P_max = 500
ALPHA = 0.90
betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ==========================================
# Helper: build parameter dicts for a scenario subset
# Re-normalizes probabilities to sum to 1
# ==========================================
def build_params(df_sub):
    scen = df_sub['Scenario_ID'].unique()
    raw  = df_sub[['Scenario_ID', 'Probability']].drop_duplicates().set_index('Scenario_ID')['Probability']
    prob = (raw / raw.sum()).to_dict()
    DA   = df_sub.set_index(['Hour', 'Scenario_ID'])['DA_Price'].to_dict()
    Bal  = df_sub.set_index(['Hour', 'Scenario_ID'])['Bal_Price'].to_dict()
    Wind = df_sub.set_index(['Hour', 'Scenario_ID'])['Wind_MW'].to_dict()
    return scen, prob, DA, Bal, Wind


# ==========================================
# Sample 200 in-sample scenarios (same seed as Task 1.3)
# ==========================================
np.random.seed(42)
selected_ids = np.random.choice(full_scenarios, 200, replace=False)
df_in = df_full[df_full['Scenario_ID'].isin(selected_ids)].copy()
scen_in, prob_in, DA_in, Bal_in, Wind_in = build_params(df_in)


# ==========================================
# One-Price CVaR Model
# ==========================================
def solve_one_price_cvar(beta, scenarios, prob, DA_price, Bal_price, Wind_MW):
    """
    Maximise: (1 - beta) * E[profit] + beta * CVaR_alpha[profit]

    Key optimisation: pre-compute scalar coefficients for P_DA[t] and the
    constant wind-revenue term per scenario outside Gurobi, then build
    constraints and objective in a single vectorised pass.  This avoids
    creating 1600 intermediate LinExpr objects (profit_w dict) which was
    the bottleneck.

    Per-scenario profit:
        pi_w = sum_t [ (DA[t,w] - BP[t,w]) * P_DA[t] + BP[t,w]*Wind[t,w] ]
             = sum_t coeff[t,w] * P_DA[t]  +  const_w

    Shortfall constraint:  lambda_w >= eta - pi_w
      =>  lambda_w - eta + sum_t coeff[t,w] * P_DA[t]  >= -const_w
    """
    m = gp.Model("OnePriceCVaR")
    m.setParam('OutputFlag', 0)

    P_DA     = m.addVars(hours, lb=0, ub=P_max, name="P_DA")
    eta      = m.addVar(lb=-GRB.INFINITY, name="eta")
    lambda_w = m.addVars(scenarios, lb=0, name="lambda")

    # Pre-compute per-scenario coefficients (pure Python, no Gurobi overhead)
    # coeff[w][t] = DA[t,w] - BP[t,w]   (net price coefficient for P_DA[t])
    # const_w     = sum_t BP[t,w] * Wind[t,w]
    coeff   = {w: {t: DA_price[t, w] - Bal_price[t, w] for t in hours} for w in scenarios}
    const_w = {w: sum(Bal_price[t, w] * Wind_MW[t, w] for t in hours)  for w in scenarios}

    # Shortfall constraints: lambda_w >= eta - pi_w
    # Rearranged: lambda_w - eta + sum_t coeff[t,w]*P_DA[t] >= -const_w
    m.addConstrs(
        (lambda_w[w] - eta + gp.quicksum(coeff[w][t] * P_DA[t] for t in hours) >= -const_w[w]
         for w in scenarios),
        name="shortfall"
    )

    # Objective coefficients for P_DA[t]:
    # E[profit] = sum_w prob[w] * (sum_t coeff[t,w]*P_DA[t] + const_w)
    #           = sum_t (sum_w prob[w]*coeff[t,w]) * P_DA[t]  +  constant
    # The constant doesn't affect optimisation so we keep it for reporting.
    exp_coeff = {t: sum(prob[w] * coeff[w][t] for w in scenarios) for t in hours}
    exp_const = sum(prob[w] * const_w[w] for w in scenarios)

    exp_profit_expr = gp.quicksum(exp_coeff[t] * P_DA[t] for t in hours)  # + exp_const (scalar)
    cvar_expr       = eta - (1.0 / (1.0 - ALPHA)) * gp.quicksum(prob[w] * lambda_w[w] for w in scenarios)

    # Full objective including the scalar constant (needed so ObjVal is correct)
    m.setObjective(
        (1 - beta) * (exp_profit_expr + exp_const) + beta * cvar_expr,
        GRB.MAXIMIZE
    )
    m.optimize()
    n_vars = m.NumVars
    n_constrs = m.NumConstrs

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"One-price CVaR model did not converge for beta={beta}")

    bids     = {t: P_DA[t].X for t in hours}
    exp_val  = sum(prob[w] * (sum(coeff[w][t] * bids[t] for t in hours) + const_w[w])
                   for w in scenarios)
    cvar_val = eta.X - (1.0 / (1.0 - ALPHA)) * sum(prob[w] * lambda_w[w].X for w in scenarios)

    return bids, exp_val, cvar_val, n_vars, n_constrs


# ==========================================
# Two-Price CVaR Model  (delta-free formulation)
# ==========================================
def solve_two_price_cvar(beta, scenarios, prob, DA_price, Bal_price, Wind_MW):
    """
    Eliminates delta_plus / delta_minus variables entirely.

    Two-price per-scenario profit written in terms of P_DA only:

        pi_w = sum_t [ DA[t,w]*Wind[t,w]                       <- scalar constant
                       + (c_plus[t,w]  - DA[t,w]) * P_DA[t]   <- when surplus (Wind > P_DA)
                       - (c_minus[t,w] - DA[t,w]) * P_DA[t] ] <- when deficit (Wind < P_DA)

    More precisely, substituting delta_plus  = max(0, Wind - P_DA)
                                   delta_minus = max(0, P_DA - Wind):

        pi_w = sum_t [ DA[t,w]*P_DA[t]
                       + c_plus[t,w]  * max(0, Wind[t,w] - P_DA[t])
                       - c_minus[t,w] * max(0, P_DA[t]  - Wind[t,w]) ]

    This is piecewise-linear in P_DA[t] with breakpoint at Wind[t,w].
    We linearise with one auxiliary variable r[t,w] >= 0:

        r[t,w] = max(0, P_DA[t] - Wind[t,w])   (the "shortage" piece)

    Then:  max(0, Wind - P_DA) = Wind - P_DA + r[t,w]

    So:
        pi_w = sum_t [ DA[t,w]*P_DA[t]
                       + c_plus[t,w]  * (Wind[t,w] - P_DA[t] + r[t,w])
                       - c_minus[t,w] * r[t,w] ]
             = sum_t [ (DA[t,w] - c_plus[t,w]) * P_DA[t]
                       + (c_plus[t,w] - c_minus[t,w]) * r[t,w]
                       + c_plus[t,w] * Wind[t,w] ]

    Constraints on r:
        r[t,w] >= P_DA[t] - Wind[t,w]   (r >= shortage)
        r[t,w] >= 0                      (already via lb=0)

    This gives only 24*1600 = 38,400 r-variables and 38,400 constraints —
    same count as delta formulation but much faster to build because there
    are no equality constraints (which Gurobi must pre-process heavily).
    """
    m = gp.Model("TwoPriceCVaR")
    m.setParam('OutputFlag', 0)

    P_DA     = m.addVars(hours, lb=0, ub=P_max, name="P_DA")
    r        = m.addVars(hours, scenarios, lb=0, name="r")   # shortage auxiliary
    eta      = m.addVar(lb=-GRB.INFINITY, name="eta")
    lambda_w = m.addVars(scenarios, lb=0, name="lambda")

    # Pre-compute scalar coefficients
    # c_plus  = min(DA, BP),  c_minus = max(DA, BP)
    c_plus  = {(t, w): min(DA_price[t, w], Bal_price[t, w]) for t in hours for w in scenarios}
    c_minus = {(t, w): max(DA_price[t, w], Bal_price[t, w]) for t in hours for w in scenarios}

    # r[t,w] >= P_DA[t] - Wind[t,w]
    m.addConstrs(
        (r[t, w] >= P_DA[t] - Wind_MW[t, w]
         for t in hours for w in scenarios),
        name="shortage_lb"
    )

    # Per-scenario profit coefficients (all scalars)
    # coeff_P[t,w]  = DA[t,w] - c_plus[t,w]
    # coeff_r[t,w]  = c_plus[t,w] - c_minus[t,w]   (always <= 0)
    # const_tw      = c_plus[t,w] * Wind[t,w]
    coeff_P = {(t, w): DA_price[t, w] - c_plus[t, w]   for t in hours for w in scenarios}
    coeff_r = {(t, w): c_plus[t, w]   - c_minus[t, w]  for t in hours for w in scenarios}
    const_w = {w: sum(c_plus[t, w] * Wind_MW[t, w] for t in hours) for w in scenarios}

    # Shortfall constraints: lambda_w >= eta - pi_w
    # => lambda_w - eta + sum_t [coeff_P*P_DA + coeff_r*r] >= -const_w
    m.addConstrs(
        (lambda_w[w] - eta
         + gp.quicksum(coeff_P[t, w] * P_DA[t] + coeff_r[t, w] * r[t, w] for t in hours)
         >= -const_w[w]
         for w in scenarios),
        name="shortfall"
    )

    # Expected profit objective coefficients aggregated over scenarios
    exp_coeff_P = {t: sum(prob[w] * coeff_P[t, w] for w in scenarios) for t in hours}
    exp_coeff_r = {(t, w): prob[w] * coeff_r[t, w] for t in hours for w in scenarios}
    exp_const   = sum(prob[w] * const_w[w] for w in scenarios)

    exp_profit_expr = (
        gp.quicksum(exp_coeff_P[t] * P_DA[t] for t in hours)
        + gp.quicksum(exp_coeff_r[t, w] * r[t, w] for t in hours for w in scenarios)
        + exp_const
    )
    cvar_expr = eta - (1.0 / (1.0 - ALPHA)) * gp.quicksum(prob[w] * lambda_w[w] for w in scenarios)

    m.setObjective((1 - beta) * exp_profit_expr + beta * cvar_expr, GRB.MAXIMIZE)
    m.optimize()

    n_vars = m.NumVars
    n_constrs = m.NumConstrs

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price CVaR model did not converge for beta={beta}")

    bids = {t: P_DA[t].X for t in hours}
    exp_val = sum(
        prob[w] * sum(
            DA_price[t, w] * bids[t]
            + c_plus[t, w]  * max(0, Wind_MW[t, w] - bids[t])
            - c_minus[t, w] * max(0, bids[t] - Wind_MW[t, w])
            for t in hours
        )
        for w in scenarios
    )
    cvar_val = eta.X - (1.0 / (1.0 - ALPHA)) * sum(prob[w] * lambda_w[w].X for w in scenarios)

    
    return bids, exp_val, cvar_val, n_vars, n_constrs


# ==========================================
# Out-of-sample evaluation helpers
# (evaluate bids on the full 1600 scenarios)
# ==========================================
def eval_oos_one_price(bids):
    return sum(
        full_prob[w] * sum(
            full_DA[t, w] * bids[t] + full_bal[t, w] * (full_wind[t, w] - bids[t])
            for t in hours
        )
        for w in full_scenarios
    )

def eval_oos_two_price(bids):
    return sum(
        full_prob[w] * sum(
            full_DA[t, w] * bids[t]
            + min(full_DA[t, w], full_bal[t, w]) * max(0, full_wind[t, w] - bids[t])
            - max(full_DA[t, w], full_bal[t, w]) * max(0, bids[t] - full_wind[t, w])
            for t in hours
        )
        for w in full_scenarios
    )


# ==========================================
# Sweep beta — optimize on 200 in-sample, evaluate on full 1600
# ==========================================
print("=" * 70)
print(f"Task 1.4 — CVaR sweep (alpha={ALPHA})  |  200 in-sample / 1600 OOS")
print("=" * 70)
print(f"{'Beta':>6} | {'InS E[P] 1p':>13} | {'OoS E[P] 1p':>13} | {'CVaR 1p':>12} | {'InS E[P] 2p':>13} | {'OoS E[P] 2p':>13} | {'CVaR 2p':>12}")
print("-" * 95)

results_1p = []
results_2p = []

loop_start = time.time()
for beta in betas:
    t0 = time.time()
    bids_1p, exp_1p_in, cvar_1p, nvars_1p, ncon_1p = solve_one_price_cvar(beta, scen_in, prob_in, DA_in, Bal_in, Wind_in)
    t1 = time.time()
    exp_1p_oos = eval_oos_one_price(bids_1p)

    t2 = time.time()
    bids_2p, exp_2p_in, cvar_2p, nvars_2p, ncon_2p = solve_two_price_cvar(beta, scen_in, prob_in, DA_in, Bal_in, Wind_in)
    t3 = time.time()
    exp_2p_oos = eval_oos_two_price(bids_2p)

    results_1p.append({'beta': beta, 'exp_in': exp_1p_in, 'exp_oos': exp_1p_oos, 'cvar': cvar_1p, 'bids': bids_1p})
    results_2p.append({'beta': beta, 'exp_in': exp_2p_in, 'exp_oos': exp_2p_oos, 'cvar': cvar_2p, 'bids': bids_2p})

    print(f"{beta:>6.2f} | {exp_1p_in:>13,.2f} | {exp_1p_oos:>13,.2f} | {cvar_1p:>12,.2f} | "
          f"{exp_2p_in:>13,.2f} | {exp_2p_oos:>13,.2f} | {cvar_2p:>12,.2f} | "
          f"1p: {t1-t0:.1f}s ({nvars_1p} vars, {ncon_1p} constr) | "
          f"2p: {t3-t2:.1f}s ({nvars_2p} vars, {ncon_2p} constr)")

print("-" * 95)
print(f"Total solver time: {time.time() - loop_start:.1f}s")

# Risk-neutral baseline = beta=0 from the same 200 in-sample scenarios
profit_neutral_1p = results_1p[0]['exp_in']
profit_neutral_2p = results_2p[0]['exp_in']


# ==========================================
# Plot 1: Expected Profit vs CVaR — efficient frontier
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, results, label, color, neutral_profit in [
    (axes[0], results_1p, "One-Price", "steelblue", profit_neutral_1p),
    (axes[1], results_2p, "Two-Price", "seagreen",  profit_neutral_2p),
]:
    exp_profits = [r['exp_in']  for r in results]
    cvars       = [r['cvar']    for r in results]

    ax.plot(cvars, exp_profits, 'o-', color=color, linewidth=2, markersize=7)

    for r in results:
        ax.annotate(
            f"β={r['beta']:.1f}",
            xy=(r['cvar'], r['exp_in']),
            xytext=(5, 3), textcoords='offset points', fontsize=10
        )

    ax.axhline(neutral_profit, color='red', linestyle='--', linewidth=1.2,
               label=f"Risk-neutral baseline (β=0): {neutral_profit:,.0f} €")

    ax.set_title(f"Task 1.4 – {label} Scheme\nExpected Profit vs CVaR$_{{α={ALPHA}}}$ (200 in-sample)", fontsize=18)
    ax.set_xlabel("CVaR (€)  [higher = less risky]", fontsize=14)
    ax.tick_params(axis='both', labelsize=12, labelrotation=45)
    ax.set_ylabel("Expected Profit (€)", fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('task1_4_efficient_frontier.png', dpi=150)
try:
    plt.show()
except KeyboardInterrupt:
    pass
plt.close()


# ==========================================
# Plot 2: In-sample vs Out-of-sample E[Profit] across beta
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, results, label, c_in, c_oos in [
    (axes[0], results_1p, "One-Price", "steelblue", "coral"),
    (axes[1], results_2p, "Two-Price", "seagreen",  "salmon"),
]:
    beta_vals = [r['beta']    for r in results]
    exp_in    = [r['exp_in']  for r in results]
    exp_oos   = [r['exp_oos'] for r in results]

    ax.plot(beta_vals, exp_in,  'o-',  color=c_in,  linewidth=2, label="In-sample (200 scen.)")
    ax.plot(beta_vals, exp_oos, 's--', color=c_oos, linewidth=2, label="Out-of-sample (1600 scen.)")

    ax.set_title(f"Task 1.4 – {label} Scheme\nIn-sample vs Out-of-sample E[Profit] across β",  fontsize=18)
    ax.set_xlabel("β (risk-aversion weight)", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("Expected Profit (€)", fontsize=14)
    ax.set_xticks(beta_vals)
    ax.legend(fontsize=18)
    ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('task1_4_oos_sensitivity.png', dpi=150)
try:
    plt.show()
except KeyboardInterrupt:
    pass
plt.close()


# ==========================================
# Plot 3: Hourly bids beta=0 vs beta=1
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, results, label, c0, c1 in [
    (axes[0], results_1p, "One-Price", "steelblue", "coral"),
    (axes[1], results_2p, "Two-Price", "seagreen",  "salmon"),
]:
    bids_b0 = results[0]['bids']
    bids_b1 = results[-1]['bids']

    ax.step(hours, [bids_b0[t] for t in hours], where='mid', color=c0,
            linewidth=2, label="β=0 (risk-neutral)")
    ax.step(hours, [bids_b1[t] for t in hours], where='mid', color=c1,
            linewidth=2, linestyle='--', label="β=1 (fully risk-averse)")

    ax.set_title(f"Task 1.4 – {label} Scheme\nHourly DA Bids: β=0 vs β=1",  fontsize=18)
    ax.set_xlabel("Hour", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("DA Bid (MW)", fontsize=14)
    ax.set_xticks(hours)
    ax.set_ylim(-10, P_max + 20)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('task1_4_bids_comparison.png', dpi=150)
plt.show()

# ==========================================
# Plot 4: Profit distribution evolution across beta (KDE, side by side)
# ==========================================
from scipy.stats import gaussian_kde

def scenario_profits_one_price(bids):
    return np.array([
        sum(
            full_DA[t, w] * bids[t] + full_bal[t, w] * (full_wind[t, w] - bids[t])
            for t in hours
        )
        for w in full_scenarios
    ])

def scenario_profits_two_price(bids):
    return np.array([
        sum(
            full_DA[t, w] * bids[t]
            + min(full_DA[t, w], full_bal[t, w]) * max(0, full_wind[t, w] - bids[t])
            - max(full_DA[t, w], full_bal[t, w]) * max(0, bids[t] - full_wind[t, w])
            for t in hours
        )
        for w in full_scenarios
    ])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.get_cmap('plasma')
beta_colors = {b: cmap(i / (len(betas) - 1)) for i, b in enumerate(betas)}

for ax, results, get_profits, label in [
    (axes[0], results_1p, scenario_profits_one_price, "One-Price"),
    (axes[1], results_2p, scenario_profits_two_price, "Two-Price"),
]:
    all_profits = np.concatenate([get_profits(r['bids']) for r in results])
    x_min, x_max = all_profits.min(), all_profits.max()
    x_grid = np.linspace(x_min, x_max, 500)

    for r in results:
        profits = get_profits(r['bids'])
        kde = gaussian_kde(profits, bw_method=0.2)
        ax.plot(x_grid, kde(x_grid), color=beta_colors[r['beta']],
                linewidth=1.8, label=f"β={r['beta']:.1f}")
        ax.axvline(profits.mean(), color=beta_colors[r['beta']],
                   linestyle=':', linewidth=0.9, alpha=0.6)

    ax.set_title(f"Task 1.4 – {label} Scheme\nProfit Distribution Evolution across β (OOS, 1600 scenarios)", fontsize=11)
    ax.set_xlabel("Scenario Profit (€)", fontsize=11)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("Density", fontsize=14 )
    ax.legend(fontsize=18, ncol=2, loc='upper right')
    ax.grid(alpha=0.3)

# sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# fig.colorbar(sm, ax=axes, label="β (risk-aversion weight)", shrink=0.7, pad=0.02)

plt.tight_layout()
plt.savefig('task1_4_profit_distribution_evolution.png', dpi=150)
try:
    plt.show()
except KeyboardInterrupt:
    pass
plt.close()

print("\nTask 1.4 complete.")
