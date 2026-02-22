import numpy as np
import pandas as pd
import os

# 1. upload data from 'scen_zone1.out' file
current_script_path = os.path.abspath(__file__)
base_dir = os.path.dirname(current_script_path)

file_path = os.path.join(base_dir, 'scen_zone1.out')  
df = pd.read_csv(file_path)

# if the CSV file has an unnamed index column, drop it
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. extract the wind power output profiles for the first 24 hours (assuming the first 24 rows correspond to the first day)

# calculate the average wind power output across all scenarios for each hour
wind_profile = df.iloc[:24, :].mean(axis=1).values

# 3. bound the wind power output between 0 and 1 (assuming the values are in per unit)
wind_profile = np.clip(wind_profile, 0.0, 1.0)

print("extracted wind power output profile for the first 24 hours:")
print(np.round(wind_profile, 4))