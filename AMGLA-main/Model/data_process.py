# ======================================
# PART 1
# Purpose: Assign different daily time windows to different sensor groups
# ======================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -------------------------------
# 1. Load data
# -------------------------------
file_path = "./PEMS04.npz"

try:
    data = np.load(file_path)
    traffic_data = data['data']
except FileNotFoundError:
    # Fallback: generate synthetic traffic data for demonstration
    num_timesteps = 1000
    num_sensors = 307
    num_features = 3

    traffic_data = np.zeros((num_timesteps, num_sensors, num_features))
    for i in range(num_sensors):
        base_flow = 50 + 40 * np.sin(
            np.linspace(0, 2 * np.pi * num_timesteps / (288 * 1.5), num_timesteps)
        )
        traffic_data[:, i, 0] = np.maximum(
            0, base_flow + np.random.randn(num_timesteps) * 10 + i * 0.1
        )
        traffic_data[:, i, 1] = np.maximum(
            10, 60 - traffic_data[:, i, 0] * 0.5 + np.random.randn(num_timesteps) * 5
        )
        traffic_data[:, i, 2] = np.maximum(
            0, traffic_data[:, i, 0] * 0.1 + np.random.randn(num_timesteps) * 2
        )

# -------------------------------
# 2. Configuration
# -------------------------------
chunk_size_i = 5        # sensors per group
chunk_size_t = 288      # time steps per day (5-min interval)
num_sensors = 307
num_time_steps = 16992  # 59 days * 288
start_base_time = datetime(2018, 1, 1)

max_groups = num_time_steps // chunk_size_t
num_groups = min((num_sensors + chunk_size_i - 1) // chunk_size_i, max_groups)

# -------------------------------
# 3. Build records
# -------------------------------
rows = []

for group_idx in range(num_groups):
    sensor_start = group_idx * chunk_size_i
    sensor_end = min(sensor_start + chunk_size_i, num_sensors)

    time_start = group_idx * chunk_size_t
    time_end = time_start + chunk_size_t

    block = traffic_data[time_start:time_end, sensor_start:sensor_end, :]
    group_start_time = start_base_time + timedelta(days=group_idx)

    for local_s in range(block.shape[1]):
        sensor_id = sensor_start + local_s
        sensor_data = block[:, local_s, :]

        for t in range(chunk_size_t):
            timestamp = group_start_time + timedelta(minutes=5 * t)
            rows.append([
                sensor_id,
                sensor_data[t, 0],
                sensor_data[t, 1],
                sensor_data[t, 2],
                timestamp
            ])

# -------------------------------
# 4. Convert to DataFrame
# -------------------------------
df = pd.DataFrame(
    rows,
    columns=['sensor_id', 'flow', 'speed', 'occupancy', 'datetime']
)
df['datetime'] = pd.to_datetime(df['datetime'])
df_p1 = df.sort_values(['sensor_id', 'datetime']).reset_index(drop=True)


# ======================================
# PART 2
# Purpose: Keep data after 03:00 each day
# ======================================

df_p2 = df_p1.copy()

# Local time index within each sensor-day
df_p2['local_t'] = df_p2.groupby('sensor_id').cumcount()

# Filter data after 03:00 (24 * 5 minutes)
df_p2 = df_p2[df_p2['local_t'] >= 24].copy()

# Re-index time and assign global index
df_p2['new_t'] = df_p2.groupby('sensor_id').cumcount()
df_p2['global_index'] = range(len(df_p2))

# Select and rename columns
df_p2['sensor_id'] += 1
df_p2 = df_p2.rename(columns={
    'sensor_id': 'prd_code',
    'datetime': 'date',
    'global_index': 'idx',
    'flow': 'OT'
})

df_p2_output = df_p2[
    ['prd_code', 'date', 'idx', 'speed', 'occupancy', 'OT']
]


# ======================================
# PART 3
# Purpose: Add unstable trend-based noise
# ======================================

trend_array = np.array([...])  # original trend values

trend_array_norm = (
    trend_array - trend_array.min()
) / (trend_array.max() - trend_array.min())

cols_to_perturb = ['speed', 'occupancy', 'OT']

def add_instable_trend_noise(group):
    n = len(group)
    x_src = np.linspace(0, 1, len(trend_array_norm))
    x_dst = np.linspace(0, 1, n)
    factors = np.interp(x_dst, x_src, trend_array_norm)

    for col in cols_to_perturb:
        group[col] = (
            group[col].values
            + factors * np.random.randn(n) * 300
            + factors * 400
        )
    return group

df_res = (
    df_p2_output
    .sort_values(['prd_code', 'date'])
    .groupby('prd_code', group_keys=False)
    .apply(add_instable_trend_noise)
)

df_res.to_csv(
    "synthetic_dataset.csv",
    index=False,
    date_format='%Y-%m-%d %H:%M'
)
