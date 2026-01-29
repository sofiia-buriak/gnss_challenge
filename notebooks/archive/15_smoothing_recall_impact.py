import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# 1. Load Data & Setup Model (Standard setup)
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Features & Target ---
SAFE_LIMIT, FAIL_LIMIT = 5000, 50000
df['degradation_score'] = ((df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)).clip(0.0, 1.0)

if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked'].replace(0, 1)
else:
    df['sat_efficiency'] = 0

features = ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency']
features = [f for f in features if f in df.columns]

# --- Train/Test Split ---
train = df[df['timestamp'] < '2025-12-01']
test = df[df['timestamp'] >= '2025-12-01'].copy() # Use copy to avoid warnings

model = XGBRegressor(objective='reg:logistic', n_jobs=-1)
model.fit(train[features], train['degradation_score'])
test['raw_pred'] = model.predict(test[features])

# ==========================================================
# 2. ANALYSIS: SMOOTHING EFFECT ON RECALL
# ==========================================================

# А. Сирий сигнал (Raw)
# Поріг 0.5. Якщо вище - тривога.
test['alert_raw'] = (test['raw_pred'] > 0.5).astype(int)

# Б. Згладжений сигнал (Smoothed)
# Rolling Mean за 10 секунд. Це "заповнить" дірки тривалістю до 10с.
window_seconds = 10
# Визначаємо вікно в рядках (приблизно)
dt = test['timestamp'].diff().dt.total_seconds().median()
if np.isnan(dt) or dt == 0: dt = 1.0
window_size = int(window_seconds / dt)

test['score_smooth'] = test['raw_pred'].rolling(window=window_size, min_periods=1).mean()
test['alert_smooth'] = (test['score_smooth'] > 0.5).astype(int)

# ==========================================================
# 3. VISUALIZATION (The "Gap Filling" Plot)
# ==========================================================
# Знайдемо фрагмент, де є "мерехтіння" (0->1->0->1)
# Шукаємо місця, де raw скаче, а smooth стабільний
diffs = test['alert_raw'].diff().abs()
noisy_idx = diffs.rolling(20).sum().idxmax() # Місце з найбільшою кількістю перемикань
if pd.notna(noisy_idx):
    center_time = test.loc[noisy_idx, 'timestamp']
    start_time = center_time - pd.Timedelta(seconds=60)
    end_time = center_time + pd.Timedelta(seconds=60)
    subset = test[(test['timestamp'] >= start_time) & (test['timestamp'] <= end_time)]
else:
    subset = test.iloc[:200]

plt.figure(figsize=(12, 8))

# Графік 1: Scores (Неперервні)
plt.subplot(2, 1, 1)
plt.plot(subset['timestamp'], subset['raw_pred'], color='lightgray', label='Raw Model Score (Noisy)', linestyle='--')
plt.plot(subset['timestamp'], subset['score_smooth'], color='blue', label=f'Smoothed Score ({window_seconds}s)', linewidth=2)
plt.axhline(0.5, color='red', linestyle=':', label='Threshold')
plt.title('Effect of Smoothing on Predicted Score')
plt.ylabel('Score (0-1)')
plt.legend()
plt.grid(True, alpha=0.3)

# Графік 2: Alerts (Бінарні - те, що бачить пілот)
plt.subplot(2, 1, 2)
# Зсуваємо графіки по вертикалі, щоб було видно різницю
plt.step(subset['timestamp'], subset['alert_raw'] + 0.1, where='post', color='gray', label='Raw Alert (Flickering)', linestyle='--')
plt.step(subset['timestamp'], subset['alert_smooth'], where='post', color='green', label='Smoothed Alert (High Recall)', linewidth=2)
plt.yticks([0, 1], ['Safe', 'Alarm'])
plt.title('Recall Improvement: Filling the Gaps')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
