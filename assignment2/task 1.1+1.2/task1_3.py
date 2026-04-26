import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os

# Import full-sample bids and expected profits from Tasks 1.1 and 1.2.
# These were optimized on all 1600 scenarios and serve as the full-sample baseline.
# NOTE: The cross-validation loop below re-optimizes on each fold's 200 in-sample
# scenarios independently — those per-fold bids cannot be imported from 1.1/1.2.
from task1_1 import run_task_1_1
from task1_2 import run_task_1_2   # assumes task1_2 exposes run_task_1_2() returning (bids, obj)

full_bids_1p, full_profit_1p = run_task_1_1()
full_bids_2p, full_profit_2p = run_task_1_2()

# ==========================================
# Task 1.3: Ex-post Analysis (8-Fold Cross-Validation)
# ==========================================
# Setup: 1600 total scenarios split into 8 folds
#   - Each fold: 200 in-sample, 1400 out-of-sample
#   - Run optimization on in-sample → evaluate on out-of-sample
#   - Do this for both one-price and two-price schemes
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'final_1600_scenarios_input.csv'
df_path = os.path.abspath(
    os.path.join(script_dir, '..', 'scenario_prep', file_name)
)

df = pd.read_csv(df_path)

hours = sorted(df['Hour'].unique())
all_scenarios = df['Scenario_ID'].unique()
P_max = 500
N_TOTAL = len(all_scenarios)   # 1600
N_IN_SAMPLE = 200
N_OUT_SAMPLE = 1400
N_FOLDS = 8

assert N_IN_SAMPLE + N_OUT_SAMPLE == N_TOTAL
assert N_TOTAL % N_FOLDS == 0  # 1600 / 8 = 200 per fold (each fold is held out once)

# ==========================================
# Helper: Build parameter dicts for a subset of scenarios
# ==========================================
def build_params(df_sub):
    """Returns (scenarios, prob, DA_price, Bal_price, Wind_MW) dicts for a scenario subset,
    with probabilities re-normalised to sum to 1."""
    scenarios = df_sub['Scenario_ID'].unique()
    raw_prob = df_sub[['Scenario_ID', 'Probability']].drop_duplicates().set_index('Scenario_ID')['Probability']
    prob = raw_prob.to_dict()
    prob_sum = raw_prob.sum()  # returned so caller can rescale profits to full probability mass
    DA_price  = df_sub.set_index(['Hour', 'Scenario_ID'])['DA_Price'].to_dict()
    Bal_price = df_sub.set_index(['Hour', 'Scenario_ID'])['Bal_Price'].to_dict()
    Wind_MW   = df_sub.set_index(['Hour', 'Scenario_ID'])['Wind_MW'].to_dict()
    return scenarios, prob, prob_sum, DA_price, Bal_price, Wind_MW


# ==========================================
# Fold-level solvers (re-optimize on 200 in-sample scenarios each fold)
# ==========================================
def solve_one_price(hours, scenarios, prob, DA_price, Bal_price, Wind_MW, model_name="OnePriceModel"):
    m = gp.Model(model_name)
    m.setParam('OutputFlag', 0)
    P_DA = m.addVars(hours, lb=0, ub=P_max, name="P_DA")
    m.setObjective(
        gp.quicksum(
            prob[w] * (DA_price[t, w] * P_DA[t] + Bal_price[t, w] * (Wind_MW[t, w] - P_DA[t]))
            for t in hours for w in scenarios
        ), GRB.MAXIMIZE
    )
    m.optimize()
    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"{model_name} did not converge (status={m.status})")
    return {t: P_DA[t].X for t in hours}, m.ObjVal


def solve_two_price(hours, scenarios, prob, DA_price, Bal_price, Wind_MW, model_name="TwoPriceModel"):
    m = gp.Model(model_name)
    m.setParam('OutputFlag', 0)
    P_DA        = m.addVars(hours, lb=0, ub=P_max, name="P_DA")
    delta_plus  = m.addVars(hours, scenarios, lb=0, name="delta_plus")
    delta_minus = m.addVars(hours, scenarios, lb=0, name="delta_minus")
    m.addConstrs(
        (Wind_MW[t, w] - P_DA[t] == delta_plus[t, w] - delta_minus[t, w]
         for t in hours for w in scenarios), name="Imbalance"
    )
    m.setObjective(
        gp.quicksum(
            prob[w] * (
                DA_price[t, w] * P_DA[t]
                + min(DA_price[t, w], Bal_price[t, w]) * delta_plus[t, w]
                - max(DA_price[t, w], Bal_price[t, w]) * delta_minus[t, w]
            )
            for t in hours for w in scenarios
        ), GRB.MAXIMIZE
    )
    m.optimize()
    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"{model_name} did not converge (status={m.status})")
    return {t: P_DA[t].X for t in hours}, m.ObjVal


# ==========================================
# Evaluation functions (no optimization — pure arithmetic)
# ==========================================
def evaluate_profit_one_price(bids, hours, scenarios, prob, DA_price, Bal_price, Wind_MW):
    return sum(
        prob[w] * (DA_price[t, w] * bids[t] + Bal_price[t, w] * (Wind_MW[t, w] - bids[t]))
        for t in hours for w in scenarios
    )


def evaluate_profit_two_price(bids, hours, scenarios, prob, DA_price, Bal_price, Wind_MW):
    total = 0.0
    for w in scenarios:
        for t in hours:
            da_p = DA_price[t, w]
            bp_p = Bal_price[t, w]
            imb  = Wind_MW[t, w] - bids[t]
            if imb >= 0:
                total += prob[w] * (da_p * bids[t] + min(da_p, bp_p) * imb)
            else:
                total += prob[w] * (da_p * bids[t] + max(da_p, bp_p) * imb)
    return total


# ==========================================
# 8-Fold Cross-Validation
# ==========================================
# Split scenario IDs into 8 contiguous blocks of 200
np.random.seed(42)
shuffled_scenarios = np.random.permutation(all_scenarios)
folds = np.array_split(shuffled_scenarios, N_FOLDS)   # 8 blocks of 200

results = []

print("=" * 70)
print("Task 1.3: 8-Fold Cross-Validation")
print(f"In-sample: {N_IN_SAMPLE} scenarios | Out-of-sample: {N_OUT_SAMPLE} scenarios")
print("=" * 70)

for fold_idx in range(N_FOLDS):
    print(f"\n--- Fold {fold_idx + 1} / {N_FOLDS} ---")

    # Split: hold out fold_idx as in-sample, rest as out-of-sample
    in_sample_ids  = folds[fold_idx]
    out_sample_ids = np.concatenate([folds[i] for i in range(N_FOLDS) if i != fold_idx])

    df_in  = df[df['Scenario_ID'].isin(in_sample_ids)]
    df_out = df[df['Scenario_ID'].isin(out_sample_ids)]

    scen_in,  prob_in,  psum_in,  DA_in,  Bal_in,  Wind_in  = build_params(df_in)
    scen_out, prob_out, psum_out, DA_out, Bal_out, Wind_out = build_params(df_out)

    # --- One-Price ---
    bids_1p, in_profit_1p = solve_one_price(
        hours, scen_in, prob_in, DA_in, Bal_in, Wind_in,
        model_name=f"OnePriceFold{fold_idx+1}"
    )
    out_profit_1p = evaluate_profit_one_price(
        bids_1p, hours, scen_out, prob_out, DA_out, Bal_out, Wind_out
    )

    # --- Two-Price ---
    bids_2p, in_profit_2p = solve_two_price(
        hours, scen_in, prob_in, DA_in, Bal_in, Wind_in,
        model_name=f"TwoPriceFold{fold_idx+1}"
    )
    out_profit_2p = evaluate_profit_two_price(
        bids_2p, hours, scen_out, prob_out, DA_out, Bal_out, Wind_out
    )

    # Rescale to full probability mass (sum = 1) so in-sample and out-of-sample
    # profits are on the same scale and directly comparable
    in_profit_1p  /= psum_in
    out_profit_1p /= psum_out
    in_profit_2p  /= psum_in
    out_profit_2p /= psum_out

    print(f"  One-Price  | In-sample: {in_profit_1p:>12,.2f} € | Out-of-sample: {out_profit_1p:>12,.2f} €")
    print(f"  Two-Price  | In-sample: {in_profit_2p:>12,.2f} € | Out-of-sample: {out_profit_2p:>12,.2f} €")

    results.append({
        'Fold': fold_idx + 1,
        'InProfit_1P':  in_profit_1p,
        'OutProfit_1P': out_profit_1p,
        'InProfit_2P':  in_profit_2p,
        'OutProfit_2P': out_profit_2p,
    })

df_results = pd.DataFrame(results)

# ==========================================
# Summary Statistics
# ==========================================
print("\n" + "=" * 70)
print("Summary across 8 folds")
print("=" * 70)

for scheme, in_col, out_col, full_profit in [
    ("One-Price", "InProfit_1P", "OutProfit_1P", full_profit_1p),
    ("Two-Price", "InProfit_2P", "OutProfit_2P", full_profit_2p),
]:
    avg_in  = df_results[in_col].mean()
    avg_out = df_results[out_col].mean()
    gap     = avg_in - avg_out
    print(f"\n{scheme}:")
    print(f"  Full-sample profit (1.1/1.2 baseline): {full_profit:>12,.2f} €")
    print(f"  Avg In-sample  profit (200 scenarios): {avg_in:>12,.2f} €")
    print(f"  Avg Out-of-sample profit (1400 scen.): {avg_out:>12,.2f} €")
    print(f"  Optimism gap (In - Out):               {gap:>12,.2f} €  ({100*gap/avg_in:.2f}%)")

# ==========================================
# Plot: In-sample vs Out-of-sample per fold
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fold_labels = [f"Fold {i+1}" for i in range(N_FOLDS)]
x = np.arange(N_FOLDS)
width = 0.35

for ax, (scheme, in_col, out_col, color_in, color_out, full_profit) in zip(axes, [
    ("One-Price Scheme", "InProfit_1P", "OutProfit_1P", "steelblue", "coral",   full_profit_1p),
    ("Two-Price Scheme", "InProfit_2P", "OutProfit_2P", "seagreen",  "salmon",  full_profit_2p),
]):
    ax.bar(x - width/2, df_results[in_col],  width, label="In-sample",     color=color_in,  alpha=0.85)
    ax.bar(x + width/2, df_results[out_col], width, label="Out-of-sample", color=color_out, alpha=0.85)

    avg_in  = df_results[in_col].mean()
    avg_out = df_results[out_col].mean()
    ax.axhline(avg_in,     color=color_in,  linestyle='--', linewidth=1.5, label=f"Avg In-sample {avg_in:,.0f} €")
    ax.axhline(avg_out,    color=color_out, linestyle='--', linewidth=1.5, label=f"Avg Out-of-sample {avg_out:,.0f} €")
    ax.axhline(full_profit, color='black',  linestyle=':',  linewidth=1.5, label=f"Full-sample baseline {full_profit:,.0f} €")

    ax.set_title(f"Task 1.3 – {scheme}\nIn-sample vs Out-of-sample Expected Profit")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Expected Profit (€)")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, rotation=30)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
#plt.savefig('task1_3_cross_validation.png', dpi=150)
plt.show()

print("\nPlot saved to task1_3_cross_validation.png")
print("Task 1.3 complete.")

# ==========================================
# Sensitivity Analysis: Optimism Gap vs In-sample Size
# 2 folds  → 800 in-sample,  800 out-of-sample
# 4 folds  → 400 in-sample, 1200 out-of-sample
# 8 folds  → 200 in-sample, 1400 out-of-sample  (already done above)
# 16 folds → 100 in-sample, 1500 out-of-sample
# ==========================================
print("\n" + "=" * 70)
print("Sensitivity Analysis: Optimism Gap vs In-sample Size")
print("=" * 70)

header = f"{'Folds':>6} | {'In-sample':>10} | {'Out-of-sample':>13} | {'Gap 1P (%)':>10} | {'Gap 2P (%)':>10}"
print(header)
print("-" * len(header))

for n_folds in [2, 4, 8, 16]:
    fold_size = N_TOTAL // n_folds
    folds_s   = np.array_split(shuffled_scenarios, n_folds)

    gaps_1p, gaps_2p = [], []

    for fold_idx in range(n_folds):
        in_ids  = folds_s[fold_idx]
        out_ids = np.concatenate([folds_s[i] for i in range(n_folds) if i != fold_idx])

        df_in  = df[df['Scenario_ID'].isin(in_ids)]
        df_out = df[df['Scenario_ID'].isin(out_ids)]

        scen_in,  prob_in,  psum_in,  DA_in,  Bal_in,  Wind_in  = build_params(df_in)
        scen_out, prob_out, psum_out, DA_out, Bal_out, Wind_out = build_params(df_out)

        bids_1p, raw_in_1p = solve_one_price(
            hours, scen_in, prob_in, DA_in, Bal_in, Wind_in,
            model_name=f"OnePriceFolds{n_folds}_F{fold_idx+1}"
        )
        raw_out_1p = evaluate_profit_one_price(bids_1p, hours, scen_out, prob_out, DA_out, Bal_out, Wind_out)

        bids_2p, raw_in_2p = solve_two_price(
            hours, scen_in, prob_in, DA_in, Bal_in, Wind_in,
            model_name=f"TwoPriceFolds{n_folds}_F{fold_idx+1}"
        )
        raw_out_2p = evaluate_profit_two_price(bids_2p, hours, scen_out, prob_out, DA_out, Bal_out, Wind_out)

        # Rescale to full probability mass before computing gap
        in_1p  = raw_in_1p  / psum_in
        out_1p = raw_out_1p / psum_out
        in_2p  = raw_in_2p  / psum_in
        out_2p = raw_out_2p / psum_out

        gaps_1p.append(100 * (in_1p - out_1p) / in_1p)
        gaps_2p.append(100 * (in_2p - out_2p) / in_2p)

    n_in  = fold_size
    n_out = N_TOTAL - fold_size
    print(f"{n_folds:>6} | {n_in:>10} | {n_out:>13} | {np.mean(gaps_1p):>10.2f} | {np.mean(gaps_2p):>10.2f}")

print("-" * len(header))
print("Interpretation: smaller optimism gap → bids generalise better to unseen scenarios.")
print("Task 1.3 complete.")