import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from collections import Counter

# ==========================================
# 1. ЗАВАНТАЖЕННЯ І ПІДГОТОВКА (як раніше)
# ==========================================
DATA_PATH = 'data/processed/all_data_compressed.parquet'
TEST_START_DATE = '2025-12-01'

print("📂 Loading data...")
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Feature Engineering (Short version) ---
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

# --- Train/Test Split & Model ---
train_df = df[df['timestamp'] < TEST_START_DATE].copy()
test_df = df[df['timestamp'] >= TEST_START_DATE].copy()

# Target
y_train = ((train_df['hAcc'] - 5000.0) / (50000.0 - 5000.0)).clip(0.0, 1.0)

print("🤖 Training Model...")
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, objective='reg:logistic', n_jobs=-1)
model.fit(train_df[features], y_train)

print("🔮 Predicting...")
test_df['predicted_score'] = model.predict(test_df[features])

# --- SMOOTHING (Critical for Rise Time) ---
# Використовуємо згладжування, щоб прибрати мікро-шум
test_df['smoothed_score'] = test_df['predicted_score'].rolling(window=5, min_periods=1).mean()

# ==========================================
# 2. АНАЛІЗ ЧАСУ НАРОСТАННЯ (RISE TIME)
# ==========================================
print("\n⏱️ Calculating Rise Times (0.1 -> 0.9)...")

rise_times = []
attack_types = [] # Instant vs Gradual

# Логіка пошуку подій
in_transition = False
start_time = None

# Для швидкості беремо only values
times = test_df['timestamp'].values
scores = test_df['smoothed_score'].values

LOW_THRESH = 0.1
HIGH_THRESH = 0.9

for i in range(1, len(scores)):
    curr_score = scores[i]
    prev_score = scores[i-1]
    curr_time = times[i]
    # 1. Початок переходу (перетнули 0.1 вгору)
    if prev_score < LOW_THRESH and curr_score >= LOW_THRESH and not in_transition:
        in_transition = True
        start_time = curr_time
    # 2. Кінець переходу (перетнули 0.9 вгору)
    if in_transition and curr_score >= HIGH_THRESH:
        duration = (curr_time - start_time) / np.timedelta64(1, 's')
        if duration > 0.5:
            rise_times.append(duration)
            if duration < 5:
                attack_types.append('Instant (<5s)')
            elif duration < 20:
                attack_types.append('Fast (5-20s)')
            else:
                attack_types.append('Gradual (>20s)')
        in_transition = False
    # 3. Скасування (якщо впали назад нижче 0.1 не дійшовши до 0.9)
    if in_transition and curr_score < LOW_THRESH:
        in_transition = False

# ==========================================
# 3. ВІЗУАЛІЗАЦІЯ І СТАТИСТИКА
# ==========================================
if len(rise_times) > 0:
    rise_times = np.array(rise_times)
    print("\n" + "="*40)
    print("📊 RISE TIME STATISTICS")
    print("="*40)
    print(f"Total Attack Events: {len(rise_times)}")
    print(f"Mean Rise Time:      {np.mean(rise_times):.2f} sec")
    print(f"Median Rise Time:    {np.median(rise_times):.2f} sec")
    print(f"Min Rise Time:       {np.min(rise_times):.2f} sec")
    print(f"Max Rise Time:       {np.max(rise_times):.2f} sec")
    print("-" * 20)
    counts = Counter(attack_types)
    for k, v in counts.items():
        print(f"{k}: {v} events ({v/len(rise_times)*100:.1f}%)")
    plt.figure(figsize=(10, 5))
    sns.histplot(rise_times, bins=20, kde=True, color='purple')
    plt.title('Distribution of Attack Rise Times (0.1 to 0.9)')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count of Attacks')
    plt.axvline(np.mean(rise_times), color='red', linestyle='--', label=f'Mean: {np.mean(rise_times):.1f}s')
    plt.legend()
    plt.show()
else:
    print("⚠️ No full transitions (0.1 -> 0.9) found in the dataset.")
