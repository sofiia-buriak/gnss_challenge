import pandas as pd
import numpy as np
import xgboost as xgb
import json
import gc
import os

# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================
BEST_PARAMS = {
    'subsample': 0.7,
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.05,
    'colsample_bytree': 0.7,
    'objective': 'reg:logistic',
    'tree_method': 'hist',
    'n_jobs': -1
}

CONFIG = {
    'model_name': 'GNSS_Early_Warning_v1',
    'input_features': [],
    'smoothing_window_seconds': 10,
    'threshold_safe': 5000,
    'threshold_fail': 50000,
    'version': '1.0.0',
    'author': 'Sofia Buriak'
}

# ==========================================
# 2. ЗАВАНТАЖЕННЯ І ПІДГОТОВКА ДАНИХ
# ==========================================
print("📂 Loading data for Final Training...")
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)

if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Target ---
print("🛠️ Re-creating features...")
t_safe = CONFIG['threshold_safe']
t_fail = CONFIG['threshold_fail']
df['degradation_score'] = ((df['hAcc'] - t_safe) / (t_fail - t_safe)).clip(0.0, 1.0)

# --- Features ---
if 'numSV' in df.columns:
    df['sat_efficiency'] = (df['numSV'] / df['numSatsTracked'].replace(0, 1)).clip(0, 5)
else:
    df['sat_efficiency'] = 0.0

# Rolling & Lag Features (Float32)
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
CONFIG['input_features'] = features

# ==========================================
# 3. SMART DOWNSAMPLING (Економія пам'яті)
# ==========================================
print("✂️ Applying Smart Downsampling...")
mask_attack = df['degradation_score'] > 0.05
mask_safe = df['degradation_score'] <= 0.05

train_attack = df[mask_attack]
train_safe = df[mask_safe].sample(frac=0.30, random_state=42)
final_train = pd.concat([train_attack, train_safe]).sample(frac=1, random_state=42)

X = final_train[features]
y = final_train['degradation_score']

del df, train_attack, train_safe, final_train
gc.collect()

# ==========================================
# 4. НАВЧАННЯ
# ==========================================
print(f"🤖 Training Final Model on {len(X)} samples...")
model = xgb.XGBRegressor(**BEST_PARAMS)
model.fit(X, y)

# ==========================================
# 5. ЗБЕРЕЖЕННЯ
# ==========================================
output_dir = 'models/production_v1'
os.makedirs(output_dir, exist_ok=True)

model_path = f"{output_dir}/gnss_model.json"
model.save_model(model_path)
print(f"✅ Model saved to: {model_path}")

config_path = f"{output_dir}/config.json"
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=4)
print(f"✅ Config saved to: {config_path}")

print("\n🎉 READY FOR DEPLOYMENT!")
print("   To load this model later:")
print("   >>> model = xgb.XGBRegressor()")
print(f"   >>> model.load_model('{model_path}')")
