import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Set up a temporary directory for PyPSA or intermediate processing
temp_dir = 'C:\\Temp_PyPSA'
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir

# ==========================================
# 1. Basic Loading and Setup (Electricity Prices)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

file_path = 'dk2_spot_prices_2024_matrix.csv'
n_clusters = 20

print("Reading and processing electricity price data...")
df_prices = pd.read_csv(file_path, index_col=0)
# Fill missing values using forward and backward fill across the hours
df_prices = df_prices.ffill(axis=1).bfill(axis=1)

# ==========================================
# 2. K-Means Clustering and Probability Calculation
# ==========================================
print(f"Extracting {n_clusters} representative price days using K-Means...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(df_prices.values)

# Identify the historical days closest to the calculated cluster centers
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df_prices.values)

# Calculate weights based on the frequency of each cluster
labels = kmeans.labels_
total_days = len(labels)
cluster_counts = Counter(labels)
probabilities = [cluster_counts[i] / total_days for i in range(n_clusters)]

# ==========================================
# 3. Data Extraction and CSV Export
# ==========================================
representative_profiles = df_prices.iloc[closest_indices]
scenarios_price = representative_profiles.values

columns = [f'Hour_{h}' for h in range(1, 25)]
index = [f'Scenario_{i+1}' for i in range(n_clusters)]
result_df = pd.DataFrame(scenarios_price, columns=columns, index=index)

# Insert metadata for easier analysis
result_df.insert(0, 'Representative_Date', representative_profiles.index.values)
result_df.insert(1, 'Cluster_ID', range(1, n_clusters + 1))
result_df.insert(2, 'Probability', probabilities)

output_file = 'price_20_scenarios_kmeans_weighted.csv'
result_df.to_csv(output_file)
print(f"Success! Weighted price CSV saved to: {output_file}")

# ==========================================
# 4. Visualization: Spaghetti Plot
# ==========================================
print("Generating price visualization charts...")
plt.figure(figsize=(14, 7))

# Background: Historical profiles for the whole year
all_historical_prices = df_prices.values
for i in range(len(all_historical_prices)):
    label = 'Historical Scenarios' if i == 0 else "_nolegend_"
    plt.plot(range(1, 25), all_historical_prices[i], color='gray', alpha=0.15, linewidth=1, label=label)

# Foreground: The 20 reduced representative scenarios
colors = cm.get_cmap('tab20', n_clusters)
for i in range(n_clusters):
    label = 'Reduced Scenarios (20 typical days)' if i == 0 else "_nolegend_"
    plt.plot(range(1, 25), scenarios_price[i], color=colors(i), linewidth=3, alpha=0.85, label=label)

plt.title('Day-Ahead Price Scenario Reduction via K-Means Clustering (DK2, 2024)', fontsize=16, fontweight='bold')
plt.xlabel('Hour of the Day', fontsize=14)
plt.ylabel('Day-Ahead Price (EUR/MWh)', fontsize=14)
plt.xticks(range(1, 25))
plt.xlim(1, 24)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')

# Export high-resolution plots
plt.savefig('Price_Scenario_Reduction_Plot.png', dpi=300, bbox_inches='tight')
plt.savefig('Price_Scenario_Reduction_Plot.pdf', bbox_inches='tight')
plt.show()

print("Charts saved successfully!")