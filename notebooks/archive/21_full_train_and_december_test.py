import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import gc

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
DATA_PATH = 'data/processed/all_data_compressed.parquet'
print("📂 Loading full dataset...")
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)

if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Зберігаємо місяць, бо timestamp ми скоро видалимо для економії
df['month'] = df['timestamp'].dt.month.astype(np.int8) # int8 економить пам'ять

print(f"   Months available: {df['month'].unique()}")

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("🛠️ Engineering features...")

# Target
SAFE_LIMIT, FAIL_LIMIT = 5000, 50000
df['degradation_score'] = ((df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)).clip(0.0, 1.0).astype(np.float32)

# Physics
if 'numSV' in df.columns:
    df['sat_efficiency'] = (df['numSV'] / df['numSatsTracked'].replace(0, 1)).clip(0, 5).astype(np.float32)
else:
    df['sat_efficiency'] = 0.0

# Rolling Features (Одразу у float32)
df = df.set_index('timestamp')
cols_to_roll = ['cnoMean', 'sat_efficiency']

for col in cols_to_roll:
    if col in df.columns:
        for w in ['5s', '10s']:
            # Використовуємо .values, щоб уникнути зайвих індексів
            roll = df[col].rolling(w)
            df[f'{col}_rolling_mean_{w}'] = roll.mean().astype(np.float32)
            df[f'{col}_rolling_std_{w}'] = roll.std().fillna(0).astype(np.float32)

df = df.reset_index()

# Lags
for col in ['cnoMean', 'sat_efficiency']:
    if col in df.columns:
        df[f'{col}_lag1'] = df[col].shift(1).bfill().astype(np.float32)

features = ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency'] + \
           [c for c in df.columns if 'rolling' in c] + \
           [c for c in df.columns if 'lag' in c]
features = [f for f in features if f in df.columns]

# ==========================================
# 3. MEMORY OPTIMIZATION (Критичний етап!)
# ==========================================
print("🧹 Aggressive Memory Cleanup...")

# 1. Залишаємо ТІЛЬКИ те, що треба для навчання. 
# Видаляємо 'timestamp', 'hAcc', 'vAcc' та всі сирі рядки
cols_to_keep = features + ['degradation_score', 'month']
df = df[cols_to_keep]

# 2. Примусова конвертація в float32 (зменшує розмір у 2 рази порівняно з float64)
float_cols = df.select_dtypes(include=['float64']).columns
if len(float_cols) > 0:
    df[float_cols] = df[float_cols].astype(np.float32)

# 3. Викликаємо збирач сміття
gc.collect()

print(f"   Dataset shape after cleanup: {df.shape}")

# ==========================================
# 4. STRICT TIME SPLIT
# ==========================================
print("\n✂️ Splitting Data by Time:")

# Спочатку робимо маску
mask_december = df['month'] == 12

# 1. Виділяємо TEST (Грудень)
X_test = df.loc[mask_december, features]
y_test = df.loc[mask_december, 'degradation_score']
print(f"   Test samples (Dec):  {len(X_test)}")

# 2. Виділяємо TRAIN (Січень-Листопад)
# Одразу фільтруємо! Не створюємо train_full
train_subset = df[~mask_december]

# Видаляємо df, щоб звільнити пам'ять під час downsampling
del df, mask_december
gc.collect()

# --- Smart Downsampling для Train ---
print("   Downsampling Train Data...")
mask_attack = train_subset['degradation_score'] > 0.05
mask_safe = train_subset['degradation_score'] <= 0.05

# Беремо 100% атак і 25% тиші
train_attack = train_subset[mask_attack]
train_safe = train_subset[mask_safe].sample(frac=0.25, random_state=42)

# Об'єднуємо
X_train = pd.concat([train_attack[features], train_safe[features]])
y_train = pd.concat([train_attack['degradation_score'], train_safe['degradation_score']])

# Перемішуємо
perm = np.random.permutation(len(X_train))
X_train = X_train.iloc[perm]
y_train = y_train.iloc[perm]

print(f"   Train samples (Opt): {len(X_train)}")

# Чистимо хвости
del train_subset, train_attack, train_safe
gc.collect()

# ==========================================
# 5. НАВЧАННЯ
# ==========================================
print("\n🤖 Training XGBoost on Past Data...")
model = xgb.XGBRegressor(
    objective='reg:logistic',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    tree_method='hist',
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Model Trained!")

# ==========================================
# 6. ПРОГНОЗ НА МАЙБУТНЄ
# ==========================================
print("🔮 Predicting December...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 RESULTS ON UNSEEN DATA (DECEMBER):")
print(f"   MAE:  {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")

# ==========================================
# 7. ВІЗУАЛІЗАЦІЯ
# ==========================================
# Шукаємо атаку в грудні для красивого графіку
y_test_np = y_test.values
subset_mask = (y_test_np > 0.1) 

if subset_mask.sum() > 0:
    idx_start = np.where(subset_mask)[0][0]
    # Беремо трохи до і трохи після атаки
    plot_slice = slice(max(0, idx_start - 200), min(len(y_test), idx_start + 800))
    print(f"\n📈 Plotting specific attack in December...")
else:
    plot_slice = slice(0, 1000)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_test))[plot_slice], y_test_np[plot_slice], label='REALITY (GPS)', color='black', alpha=0.5)
plt.plot(np.arange(len(y_test))[plot_slice], y_pred[plot_slice], label='FORECAST (Model)', color='red', linewidth=2, alpha=0.8)

# Згладжування для візуалізації
smooth_pred = pd.Series(y_pred[plot_slice]).rolling(10, min_periods=1).mean()
plt.plot(np.arange(len(y_test))[plot_slice], smooth_pred, label='Smoothed Output', color='blue', linewidth=2)

plt.title('Validation on Future Data (December)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()