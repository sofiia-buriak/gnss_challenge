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

# Feature engineering: sat_efficiency
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked']
    df['sat_efficiency'] = df['sat_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    df['sat_efficiency'] = 0

# 2. Event Extraction (Bidirectional)
label = df['overallPositionLabel'].values
onset_idx = np.where((label[:-1] == 0) & (label[1:] == 1))[0] + 1  # 0->1
recovery_idx = np.where((label[:-1] == 1) & (label[1:] == 0))[0] + 1  # 1->0

window_size = 30  # seconds before and after

# Helper to extract windows
windows_onset = []
windows_recovery = []
for idx in onset_idx:
    if idx - window_size >= 0 and idx + window_size < len(df):
        win = df.iloc[idx-window_size:idx+window_size+1][['sat_efficiency', 'cnoMean']].reset_index(drop=True)
        win['rel_time'] = np.arange(-window_size, window_size+1)
        windows_onset.append(win)
for idx in recovery_idx:
    if idx - window_size >= 0 and idx + window_size < len(df):
        win = df.iloc[idx-window_size:idx+window_size+1][['sat_efficiency', 'cnoMean']].reset_index(drop=True)
        win['rel_time'] = np.arange(-window_size, window_size+1)
        windows_recovery.append(win)

# 3. Visualization (The Hysteresis Loop)
sns.set_style('whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Onset (0->1)
for win in windows_onset:
    axes[0].plot(win['rel_time'], win['sat_efficiency'], color='blue', alpha=0.1, linewidth=1)
mean_onset = pd.concat(windows_onset).groupby('rel_time').mean()
axes[0].plot(mean_onset.index, mean_onset['sat_efficiency'], color='blue', linewidth=3, label='Mean')
axes[0].set_title('Attack Onset (0→1)')
axes[0].set_xlabel('Time (s, 0=Transition)')
axes[0].set_ylabel('sat_efficiency')
axes[0].legend()

# Recovery (1->0)
for win in windows_recovery:
    axes[1].plot(win['rel_time'], win['sat_efficiency'], color='green', alpha=0.1, linewidth=1)
mean_recovery = pd.concat(windows_recovery).groupby('rel_time').mean()
axes[1].plot(mean_recovery.index, mean_recovery['sat_efficiency'], color='green', linewidth=3, label='Mean')
axes[1].set_title('Recovery (1→0)')
axes[1].set_xlabel('Time (s, 0=Transition)')
axes[1].legend()

plt.suptitle('Hysteresis Loop: Satellite Efficiency Around Transitions', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 4. Slope Calculation (Velocity)
def avg_slope(windows, metric):
    slopes = []
    for win in windows:
        y = win[metric].values
        # Slope from t=-5 to t=0 (5 points)
        slope = (y[window_size] - y[window_size-5]) / 5
        slopes.append(slope)
    return np.mean(slopes), np.std(slopes)

onset_slope, onset_std = avg_slope(windows_onset, 'sat_efficiency')
recovery_slope, recovery_std = avg_slope(windows_recovery, 'sat_efficiency')

print(f"Average Time to Fail (sat_efficiency, 5s slope): {onset_slope:.4f} ± {onset_std:.4f}")
print(f"Average Time to Recover (sat_efficiency, 5s slope): {recovery_slope:.4f} ± {recovery_std:.4f}")

if abs(recovery_slope) > abs(onset_slope):
    print('Recovery is FASTER than Failure (steeper slope).')
elif abs(recovery_slope) < abs(onset_slope):
    print('Failure is FASTER than Recovery (steeper slope).')
else:
    print('Failure and Recovery have similar rates.')

# (Optional) Repeat for cnoMean
onset_slope_cno, _ = avg_slope(windows_onset, 'cnoMean')
recovery_slope_cno, _ = avg_slope(windows_recovery, 'cnoMean')
print(f"\n[Signal Strength] Average Time to Fail (cnoMean, 5s slope): {onset_slope_cno:.4f}")
print(f"[Signal Strength] Average Time to Recover (cnoMean, 5s slope): {recovery_slope_cno:.4f}")
