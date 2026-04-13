import pandas as pd
import numpy as np
import itertools
import os

# Set up temporary directory for processing
temp_dir = 'C:\\Temp_PyPSA'
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir

print("Starting the generation of the final 1,600 joint scenarios...")

# ==========================================
# 1. Load Weighted Wind and Price Scenarios
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load wind power scenarios
df_wind = pd.read_csv('wind_20_scenarios_kmeans_weighted_500MW.csv', index_col=0)
# Extract probability column and 24-hour generation values (Hour_1 to Hour_24)
wind_probs = df_wind['Probability'].values
wind_values = df_wind.loc[:, 'Hour_1':'Hour_24'].values  # Shape: (20, 24)

# Load electricity price scenarios
df_price = pd.read_csv('price_20_scenarios_kmeans_weighted.csv', index_col=0)
price_probs = df_price['Probability'].values
price_values = df_price.loc[:, 'Hour_1':'Hour_24'].values # Shape: (20, 24)

# ==========================================
# 2. Generate 4 System Imbalance (SI) Scenarios
# ==========================================
# 0 represents Surplus, 1 represents Deficit
num_si = 4
np.random.seed(42) # Fix random seed for reproducibility
# Generate 4 random 24-hour state sequences (50% probability for 0 or 1)
si_values = np.random.binomial(1, 0.5, size=(num_si, 24))

# Assuming these 4 sequences are generated with equal probability (1/4 = 0.25)
si_probs = [1.0 / num_si] * num_si

# ==========================================
# 3. Cartesian Product Combination (20 x 20 x 4 = 1600)
# ==========================================
num_wind = len(wind_values)
num_price = len(price_values)

# Initialize a list to store all data records
all_records = []
total_prob_check = 0.0 # Used to validate that total joint probability equals 1.0

# Iterate through all possible combinations
scenario_id = 1
for w_idx, p_idx, s_idx in itertools.product(range(num_wind), range(num_price), range(num_si)):
    
    # Get 24-hour profile data for the current combination
    w_24h = wind_values[w_idx]
    p_24h = price_values[p_idx]
    s_24h = si_values[s_idx]
    
    # Calculate joint probability: P_joint = P_wind * P_price * P_si
    prob_joint = wind_probs[w_idx] * price_probs[p_idx] * si_probs[s_idx]
    total_prob_check += prob_joint
    
    # Calculate Balancing Price (BP) for each hour of the day
    for h in range(24):
        # Calculate BP based on assignment rules:
        # If SI=1 (deficit), BP = 1.25 * DA price
        # If SI=0 (surplus), BP = 0.85 * DA price
        if s_24h[h] == 1:
            bp = 1.25 * p_24h[h]
        else:
            bp = 0.85 * p_24h[h]
            
        # Append record to list
        all_records.append({
            'Scenario_ID': scenario_id,
            'Probability': prob_joint,
            'Hour': h + 1,
            'Wind_MW': w_24h[h],
            'DA_Price': p_24h[h],
            'SI_State': s_24h[h],
            'Bal_Price': bp
        })
        
    scenario_id += 1

# ==========================================
# 4. Export and Validation
# ==========================================
df_final = pd.DataFrame(all_records)

# Export as the final input table
output_file = 'final_1600_scenarios_input.csv'
df_final.to_csv(output_file, index=False)

print(f"\nSuccess! The 1,600 joint scenarios table has been generated: {output_file}")
print(f"Total rows: {len(df_final)} (1,600 scenarios * 24 hours)")
print(f"Joint probability check (sum should be 1.0): {total_prob_check:.6f}")
print("\nData Preview (First 5 rows):")
print(df_final.head())