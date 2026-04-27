import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Set random seed for reproducibility
np.random.seed(42)

# 2. Define parameters given in the assignment
n_total = 300           # Total number of profiles
n_minutes = 60          # 60-minute resolution for one hour
p_min, p_max = 220, 600 # Power consumption range (kW) 
ramp_limit = 35         # Maximum variation between minutes (kW) 

# 3. Generate random load profiles
profiles = np.zeros((n_total, n_minutes))

for s in range(n_total):
    # Randomly generate the initial value within the range
    profiles[s, 0] = np.random.uniform(p_min, p_max)
    
    for t in range(1, n_minutes):
        # Generate random fluctuations satisfying the ramp constraint
        delta = np.random.uniform(-ramp_limit, ramp_limit)
        next_val = profiles[s, t-1] + delta
        
        # Apply boundary constraints [220, 600]
        profiles[s, t] = np.clip(next_val, p_min, p_max)

# save results as .csv
# 1. create columns (Minute_0, Minute_1, ..., Minute_59)
columns = [f'Minute_{i}' for i in range(n_minutes)]

# 2. transform to DataFrame，and add Scenario index
df = pd.DataFrame(profiles, columns=columns)
df.index.name = 'Scenario_ID'

# 3. output CSV file
# index=True ->reserve Scenario_ID，to locate In-sample (0-99) and Out-of-sample (100-299)
file_name = 'stochastic_load_profiles.csv'
df.to_csv(file_name)

print(f"Successfully export {n_total} scenarios to file: {file_name}")

# 4. Dataset splitting 
in_sample = profiles[:100, :]
out_of_sample = profiles[100:, :]

# 5. Statistical check (for written report description)
global_mean = np.mean(profiles)
in_sample_mean = np.mean(in_sample)
out_sample_mean = np.mean(out_of_sample)

print(f"--- Load Scenario Statistics ---")
print(f"Global average load: {global_mean:.2f} kW")
print(f"In-sample average load: {in_sample_mean:.2f} kW")
print(f"Out-of-sample average load: {out_sample_mean:.2f} kW")
print(f"Verify ramp constraints: {'Pass' if np.all(np.abs(np.diff(profiles, axis=1)) <= ramp_limit + 1e-9) else 'Fail'}")

# 6. Plotting (for report figures)
plt.figure(figsize=(10, 6))
# Plot the first 5 curves as examples
for i in range(5):
    plt.plot(profiles[i], label=f'Scenario {i+1}', alpha=0.8)

# Plot boundary lines
plt.axhline(y=p_min, color='r', linestyle='--', label='Lower Bound (220 kW)')
plt.axhline(y=p_max, color='g', linestyle='--', label='Upper Bound (600 kW)')

plt.title('Generated Stochastic Load Profiles (Step 2)', fontsize=14)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Consumption (kW)', fontsize=12)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()

# plt.savefig('load_profiles.png', dpi=300)
plt.show()
