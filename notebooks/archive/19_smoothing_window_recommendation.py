import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА (АВТОНОМНО)
# ==========================================
DATA_PATH = 'data/processed/all_data_compressed.parquet'
print("📂 Loading data...")
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Target
SAFE_LIMIT, FAIL_LIMIT = 5000, 50000
df['degradation_score'] = ((df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)).clip(0.0, 1.0)

# Features
if 'numSV' in df.columns:
    df['sat_efficiency'] = (df['numSV'] / df['numSatsTracked'].replace(0, 1)).clip(0, 5)
else:
    df['sat_efficiency'] = 0.0

df = df.set_index('timestamp')
for col in ['cnoMean', 'sat_efficiency']:
    if col in df.columns:
        for w in ['5s', '10s']:
            df[f'{col}_rolling_mean_{w}'] = df[col].rolling(w).mean().astype(np.float32)
            df[f'{col}_rolling_std_{w}'] = df[col].rolling(w).std().fillna(0).astype(np.float32)
df = df.reset_index()

for col in ['cnoMean', 'sat_efficiency']:
    if col in df.columns:
        df[f'{col}_lag1'] = df[col].shift(1).bfill().astype(np.float32)

features = ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency'] + \
           [c for c in df.columns if 'rolling' in c] + \
           [c for c in df.columns if 'lag' in c]
features = [f for f in features if f in df.columns]

# ==========================================
# 2. ШВИДКИЙ ПЕРЕРАХУНОК МОДЕЛІ
# ==========================================
print("🤖 Quick Model Retrain (to get raw predictions)...")
split_date = pd.Timestamp('2025-12-01')
# Беремо маленьку вибірку для швидкості (нам треба лише тренд)
train = df[df['timestamp'] < split_date].sample(frac=0.2, random_state=42)
test = df[df['timestamp'] >= split_date].copy()

model = XGBRegressor(
    objective='reg:logistic', 
    n_jobs=-1, 
    tree_method='hist',
    n_estimators=100, 
    max_depth=6
)
model.fit(train[features], train['degradation_score'])

print("🔮 Predicting...")
test['pred_raw'] = model.predict(test[features])

# ==========================================
# 3. ПОШУК ЗОНИ ХАОСУ (Re-Find Chaos Zone)
# ==========================================
print("🔍 Re-locating the chaos zone...")
# Шукаємо де сигнал стрибав найсильніше
test['volatility'] = test['pred_raw'].diff().abs().rolling(60).sum()
chaos_idx = test['volatility'].idxmax()
chaos_time = test.loc[chaos_idx, 'timestamp']

print(f"📍 Chaos found at: {chaos_time}")

# Вирізаємо вікно 4 хвилини
start_t = chaos_time - pd.Timedelta(seconds=120)
end_t = chaos_time + pd.Timedelta(seconds=120)
subset = test[(test['timestamp'] >= start_t) & (test['timestamp'] <= end_t)].copy()

# ==========================================
# 4. ПІДБІР ВІКНА (GRID SEARCH)
# ==========================================
print("\n🧪 TESTING WINDOW SIZES:")
windows = [10, 30, 60, 90, 120]
results = []
colors = ['red', 'orange', 'gold', 'green', 'blue']

plt.figure(figsize=(14, 8))
plt.plot(subset['timestamp'], subset['pred_raw'], color='lightgray', label='Raw Model Output', alpha=0.6)

for i, w in enumerate(windows):
    col_name = f'smooth_{w}s'
    # Рахуємо середнє
    subset[col_name] = subset['pred_raw'].rolling(window=w, min_periods=1).mean()
    # Рахуємо стабільність (чим менше std, тим краще)
    stability = subset[col_name].std()
    
    status = "✅ ACCEPTABLE" if stability < 0.1 else "❌ TOO NOISY"
    print(f"   Window {w}s -> Std Dev: {stability:.4f} | {status}")
    
    results.append({'window': w, 'stability': stability})
    
    plt.plot(subset['timestamp'], subset[col_name], 
             label=f'Window {w}s ($\sigma$={stability:.2f})', 
             linewidth=2, color=colors[i])

plt.title(f'Smoothing Window Optimization\nChaos Event at {chaos_time}')
plt.ylabel('Score Stability')
plt.xlabel('Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Рекомендація
best_win = None
for res in results:
    if res['stability'] < 0.1:
        best_win = res['window']
        break

print("-" * 40)
if best_win:
    print(f"🏆 FINAL RECOMMENDATION: Use a **{best_win}-second** Moving Average.")
    print(f"   Reason: It's the smallest window that keeps noise below 0.1.")
else:
    print(f"⚠️ RECOMMENDATION: Use **120+ seconds** or a Hysteresis Filter.")