import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and sort data
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Session Identification (Reconstruct Missions)
df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
df['session_id'] = (df['time_diff'] > 60).cumsum()

# 3. Calculate Time Since Start (minutes)
session_starts = df.groupby('session_id')['timestamp'].min()
df['t_start'] = df['session_id'].map(session_starts)
df['time_since_start'] = (df['timestamp'] - df['t_start']).dt.total_seconds() / 60.0

# 4. Identify Moment of Death (First Failure)
first_failures = df[df['overallPositionLabel'] == 1].groupby('session_id').first().reset_index()
# Filter out sessions where first row is a failure (dead on arrival)
session_first_rows = df.groupby('session_id').first().reset_index()
dead_on_arrival_sessions = session_first_rows[session_first_rows['overallPositionLabel'] == 1]['session_id'].tolist()
first_failures = first_failures[~first_failures['session_id'].isin(dead_on_arrival_sessions)]

# Extract TTFF (Time to First Failure)
ttff_minutes = first_failures['time_since_start']

# 5. Visualization
plt.figure(figsize=(10,6))
sns.histplot(ttff_minutes, bins=np.arange(0, ttff_minutes.max()+5, 5), kde=False, color='crimson', edgecolor='black')
plt.xlabel('Time to First Failure (Minutes)')
plt.ylabel('Number of Sessions')
plt.title('Distribution of Time to First Failure (TTFF)')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary stats
print(f"Total sessions: {df['session_id'].nunique()}")
print(f"Sessions with at least one failure: {len(ttff_minutes)}")
print(f"Dead on arrival sessions: {len(dead_on_arrival_sessions)}")
print(f"Median TTFF (min): {ttff_minutes.median():.2f}")
print(f"Mean TTFF (min): {ttff_minutes.mean():.2f}")
print(f"TTFF < 5 min: {(ttff_minutes < 5).sum()}")
print(f"TTFF < 15 min: {(ttff_minutes < 15).sum()}")
