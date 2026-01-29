import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# ==========================================
# 1. ШВИДКЕ НАЛАШТУВАННЯ (ЯКЩО ЗАПУСКАЄШ ОКРЕМО)
# ==========================================
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Target & Features
SAFE_LIMIT, FAIL_LIMIT = 5000, 50000
df['degradation_score'] = ((df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)).clip(0.0, 1.0)

if 'numSV' in df.columns:
    df['sat_efficiency'] = (df['numSV'] / df['numSatsTracked'].replace(0, 1)).clip(0, 5)
else:
    df['sat_efficiency'] = 0.0

# Rolling Features (float32)
df = df.set_index('timestamp')
for col in ['cnoMean', 'sat_efficiency']:
    if col in df.columns:
        for w in ['5s', '10s']:
            df[f'{col}_rolling_mean_{w}'] = df[col].rolling(w).mean().astype(np.float32)
            df[f'{col}_rolling_std_{w}'] = df[col].rolling(w).std().fillna(0).astype(np.float32)
df = df.reset_index()

# Lags
for col in ['cnoMean', 'sat_efficiency']:
    if col in df.columns:
        df[f'{col}_lag1'] = df[col].shift(1).bfill().astype(np.float32)

features = ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency'] + \
           [c for c in df.columns if 'rolling' in c] + \
           [c for c in df.columns if 'lag' in c]
features = [f for f in features if f in df.columns]

# Split
date_split = pd.Timestamp('2025-12-01')
train_df = df[df['timestamp'] < date_split].sample(frac=0.25, random_state=42) # Downsample for speed
test_df = df[df['timestamp'] >= date_split].copy()

# Train (Light)
model = XGBRegressor(objective='reg:logistic', n_jobs=-1, tree_method='hist', max_depth=6, n_estimators=100)
model.fit(train_df[features], train_df['degradation_score'])

# Predict
test_df['pred_raw'] = model.predict(test_df[features])

# ==========================================
# 2. ПОШУК ЗОНИ "МЕРЕХТІННЯ" (CHATTERING HUNT)
# ==========================================
print("🔍 Hunting for the most chaotic segment...")

test_df['volatility'] = test_df['degradation_score'].diff().abs().rolling(window=60).sum()
chaos_idx = test_df['volatility'].idxmax()
chaos_time = test_df.loc[chaos_idx, 'timestamp']
print(f"📍 Most chaotic moment found at: {chaos_time}")

window_seconds = 120 
start_time = chaos_time - pd.Timedelta(seconds=window_seconds)
end_time = chaos_time + pd.Timedelta(seconds=window_seconds)
subset = test_df[(test_df['timestamp'] >= start_time) & (test_df['timestamp'] <= end_time)].copy()

# ==========================================
# 3. ВІЗУАЛІЗАЦІЯ: БИТВА РЕАЛЬНОСТІ ПРОТИ МОДЕЛІ
# ==========================================
plt.figure(figsize=(14, 7))
plt.plot(subset['timestamp'], subset['degradation_score'], 
         color='black', alpha=0.3, linewidth=1, label='Actual GPS (Chattering Target)')
plt.plot(subset['timestamp'], subset['pred_raw'], 
         color='red', linewidth=2, alpha=0.8, label='Physics Model Prediction (Raw)')
subset['pred_smooth'] = subset['pred_raw'].rolling(window=10, min_periods=1).mean()
plt.plot(subset['timestamp'], subset['pred_smooth'], 
         color='blue', linewidth=3, label='Smoothed Output (For Pilot)')
plt.title(f'Stress Test: Model Behavior during Signal Flickering\nCentered at {chaos_time}')
plt.ylabel('Degradation Score (0=Good, 1=Bad)')
plt.xlabel('Time')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.axhline(0.9, color='red', linestyle=':', alpha=0.5)
plt.text(subset['timestamp'].iloc[0], 0.92, 'CRITICAL ZONE', color='red', fontsize=8)
plt.tight_layout()
plt.show()

# ==========================================
# 4. ЛОКАЛЬНА ОЦІНКА ПОМИЛКИ
# ==========================================
mse_chaos = np.mean((subset['degradation_score'] - subset['pred_raw'])**2)
rmse_chaos = np.sqrt(mse_chaos)

print("\n📊 Local Statistics (Inside the Chaos Zone):")
print(f"   RMSE during flickering: {rmse_chaos:.4f}")
print(f"   Model Stability (Std Dev): {subset['pred_raw'].std():.4f}")
print(f"   Target Instability (Std Dev): {subset['degradation_score'].std():.4f}")


# Рахуємо стабільність вже ЗГЛАДЖЕНОГО сигналу
stability_smooth = subset['pred_smooth'].std()
print(f"   Smoothed Output Stability: {stability_smooth:.4f}")

if stability_smooth < 0.1:
    print("✅ FINAL VERDICT: Rolling Filter successfully tamed the noise.")
else:
    print("⚠️ Warning: Even smoothing didn't help (Need larger window?)")
