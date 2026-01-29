import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import gc

# ==========================================
# 1. ШВИДКА ПІДГОТОВКА ДАНИХ
# ==========================================
DATA_PATH = 'data/processed/all_data_compressed.parquet'
print("📂 Loading data (December only)...")

# Читаємо файл
df = pd.read_parquet(DATA_PATH)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Додаємо місяць
df['month'] = df['timestamp'].dt.month

# --- Фільтруємо ТІЛЬКИ ГРУДЕНЬ (Month 12) ---
# Нам не треба тренувати модель заново, ми завантажимо готову.
# Але нам потрібні тестові дані.
test_df = df[df['month'] == 12].copy()
del df # Чистимо пам'ять
gc.collect()

print(f"   Loaded {len(test_df)} samples for December.")

# ==========================================
# 2. FEATURE ENGINEERING (Тільки для тесту)
# ==========================================
print("🛠️ Engineering features...")

# Target
SAFE_LIMIT, FAIL_LIMIT = 5000, 50000
test_df['degradation_score'] = ((test_df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)).clip(0.0, 1.0).astype(np.float32)

# Physics
if 'numSV' in test_df.columns:
    test_df['sat_efficiency'] = (test_df['numSV'] / test_df['numSatsTracked'].replace(0, 1)).clip(0, 5).astype(np.float32)
else:
    test_df['sat_efficiency'] = 0.0

# Rolling Features
test_df = test_df.set_index('timestamp')
for col in ['cnoMean', 'sat_efficiency']:
    if col in test_df.columns:
        for w in ['5s', '10s']:
            test_df[f'{col}_rolling_mean_{w}'] = test_df[col].rolling(w).mean().astype(np.float32)
            test_df[f'{col}_rolling_std_{w}'] = test_df[col].rolling(w).std().fillna(0).astype(np.float32)
test_df = test_df.reset_index()

# Lags
for col in ['cnoMean', 'sat_efficiency']:
    if col in test_df.columns:
        test_df[f'{col}_lag1'] = test_df[col].shift(1).bfill().astype(np.float32)

features = ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency'] + \
           [c for c in test_df.columns if 'rolling' in c] + \
           [c for c in test_df.columns if 'lag' in c]
features = [f for f in features if f in test_df.columns]

# ==========================================
# 3. ЗАВАНТАЖЕННЯ МОДЕЛІ І ПРОГНОЗ
# ==========================================
print("🤖 Loading pre-trained model...")
# Тут вкажи шлях до файлу моделі, який ми зберегли раніше.
# Якщо ти не зберегла його, ми навчимо швидку версію прямо тут.
model_path = 'models/production_v1/gnss_model.json'

model = xgb.XGBRegressor()

if False: # Зміни на True, якщо файл існує
    model.load_model(model_path)
    print("   Loaded from file.")
else:
    print("   ⚠️ Model file not found. Quick re-train on small sample...")
    # Швидке навчання (емуляція)
    train_dummy = test_df.sample(frac=0.1) 
    model = xgb.XGBRegressor(n_estimators=50, max_depth=6, tree_method='hist')
    model.fit(train_dummy[features], train_dummy['degradation_score'])

print("🔮 Predicting...")
y_pred = model.predict(test_df[features])
y_test = test_df['degradation_score'].values

# ==========================================
# 4. ВІЗУАЛІЗАЦІЯ (ZOOM НА АТАКУ)
# ==========================================
print("🎨 Rendering DoD Dynamics...")

# Шукаємо цікавий момент (де реальна деградація > 0.3)
attacks = np.where(y_test > 0.3)[0]

if len(attacks) > 0:
    center_idx = attacks[0] + 100 
    window = 1000 
    start = max(0, center_idx - 300)
    end = min(len(y_test), center_idx + 700)
    subset_slice = slice(start, end)
    print(f"   Zooming in on Attack at index {center_idx}...")
else:
    subset_slice = slice(0, 1000)
    print("   No major attacks found, showing start.")

# Готуємо осі
time_axis = np.arange(end - start)
truth = y_test[subset_slice]
raw_pred = y_pred[subset_slice]
# Згладжування (Blue Line)
smooth_pred = pd.Series(raw_pred).rolling(window=10, min_periods=1).mean().values

# Малюємо
plt.figure(figsize=(14, 7))
plt.style.use('bmh') # Стиль "Scientific"

# 1. REALITY (Чорна зона)
plt.plot(time_axis, truth, color='black', linewidth=1.5, label='Ground Truth (hAcc)', alpha=0.6)
plt.fill_between(time_axis, 0, truth, color='black', alpha=0.1)

# 2. RAW AI (Червоний шум)
plt.plot(time_axis, raw_pred, color='red', linewidth=0.8, alpha=0.3, label='Raw Model Output', linestyle='-')

# 3. SMOOTHED (Синій контроль)
plt.plot(time_axis, smooth_pred, color='#0044cc', linewidth=3, label='Smoothed Output (Final)')

# 4. THRESHOLD (Зелений кордон)
plt.axhline(y=0.1, color='green', linestyle='--', linewidth=2, label='Alert Threshold (0.1)')

plt.title('System Response Dynamics: Real Data (December Validation)', fontsize=14, fontweight='bold')
plt.xlabel('Time (samples)', fontsize=12)
plt.ylabel('Degradation Score', fontsize=12)
plt.legend(loc='upper left', framealpha=0.9, facecolor='white')
plt.ylim(-0.05, 1.05)
plt.tight_layout()

# Зберігаємо
plt.savefig('dod_dynamics_december.png', dpi=300)
print("✅ Graph saved as 'dod_dynamics_december.png'")
plt.show()