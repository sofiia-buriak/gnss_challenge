import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from collections import Counter

# 1. Setup & Prediction
DATA_PATH = 'data/processed/all_data_compressed.parquet'
TEST_START_DATE = '2025-12-01'

print("📂 Loading data...")
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Physics-only features
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked'].replace(0, 1)
    df['sat_efficiency'] = df['sat_efficiency'].clip(0, 5)
else:
    df['sat_efficiency'] = 0.0
for col in ['cnoMean', 'sat_efficiency', 'numSV']:
    if col in df.columns:
        df[f'{col}_lag1'] = df[col].shift(1).bfill()
features = ['cnoMean', 'cnoMean_lag1', 'sat_efficiency', 'sat_efficiency_lag1', 'numSV', 'numSV_lag1', 'numSatsTracked', 'cnoStd']
features = [f for f in features if f in df.columns]

# Target engineering
SAFE_LIMIT = 5000
FAIL_LIMIT = 50000
df['degradation_score'] = (df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)
df['degradation_score'] = df['degradation_score'].clip(0.0, 1.0)

# Train/test split
train_df = df[df['timestamp'] < TEST_START_DATE].copy()
test_df = df[df['timestamp'] >= TEST_START_DATE].copy()

print("🤖 Training Model...")
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, objective='reg:logistic', n_jobs=-1)
model.fit(train_df[features], train_df['degradation_score'])

print("🔮 Predicting...")
test_df['predicted_score'] = model.predict(test_df[features])

# Smoothing
print("🧹 Smoothing predicted score...")
test_df['smoothed_score'] = test_df['predicted_score'].rolling(window=5, min_periods=1).mean()

# 2. Rise Time Calculation Algorithm
print("\n⏱️ Calculating Rise Times (0.1 -> 0.9)...")
rise_times = []
rise_indices = []
attack_types = []

in_transition = False
start_time = None
start_idx = None

times = test_df['timestamp'].values
scores = test_df['smoothed_score'].values

LOW_THRESH = 0.1
HIGH_THRESH = 0.9

for i in range(1, len(scores)):
    curr_score = scores[i]
    prev_score = scores[i-1]
    curr_time = times[i]
    # Start event
    if prev_score < LOW_THRESH and curr_score >= LOW_THRESH and not in_transition:
        in_transition = True
        start_time = curr_time
        start_idx = i
    # End event
    if in_transition and curr_score >= HIGH_THRESH:
        duration = (curr_time - start_time) / np.timedelta64(1, 's')
        if duration > 0.5:
            rise_times.append(duration)
            rise_indices.append((start_idx, i))
            if duration < 5:
                attack_types.append('Instant (<5s)')
            elif duration < 30:
                attack_types.append('Fast (5-30s)')
            else:
                attack_types.append('Gradual (>30s)')
        in_transition = False
    # Cancel event if drops below 0.1
    if in_transition and curr_score < LOW_THRESH:
        in_transition = False

# 3. Statistical Analysis
if len(rise_times) > 0:
    rise_times = np.array(rise_times)
    print("\n" + "="*40)
    print("📊 RISE TIME STATISTICS")
    print("="*40)
    print(f"Total Events: {len(rise_times)}")
    print(f"Median Rise Time: {np.median(rise_times):.2f} sec")
    print(f"Min Rise Time:    {np.min(rise_times):.2f} sec")
    print(f"Max Rise Time:    {np.max(rise_times):.2f} sec")
    print("-" * 20)
    counts = Counter(attack_types)
    for k, v in counts.items():
        print(f"{k}: {v} events ({v/len(rise_times)*100:.1f}%)")
else:
    print("⚠️ No full transitions (0.1 -> 0.9) found in the dataset.")

# 4. Visualization
if len(rise_times) > 0:
    plt.figure(figsize=(10, 5))
    sns.histplot(rise_times, bins=20, kde=True, color='purple')
    plt.title('Distribution of Rise Times (0.1 to 0.9)')
    plt.xlabel('Rise Time (seconds)')
    plt.ylabel('Count')
    plt.axvline(np.median(rise_times), color='red', linestyle='--', label=f'Median: {np.median(rise_times):.1f}s')
    plt.legend()
    plt.show()
    # Zoomed transition plot for a "Gradual" event
    if 'Gradual (>30s)' in attack_types:
        idx = attack_types.index('Gradual (>30s)')
        start, end = rise_indices[idx]
        plt.figure(figsize=(12, 4))
        plt.plot(test_df['timestamp'].iloc[start-10:end+10], test_df['smoothed_score'].iloc[start-10:end+10], color='blue')
        plt.axhline(0.1, color='green', linestyle='--', label='0.1 Threshold')
        plt.axhline(0.9, color='red', linestyle='--', label='0.9 Threshold')
        plt.title('Zoomed Transition: Gradual Event')
        plt.xlabel('Timestamp')
        plt.ylabel('Smoothed Score')
        plt.legend()
        plt.tight_layout()
        plt.show()
