import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import gc

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
DATA_PATH = 'data/processed/all_data_compressed.parquet'
print("📂 Loading data...")

df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)

if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("🛠️ Engineering features...")

# Target
SAFE_LIMIT, FAIL_LIMIT = 5000, 50000
df['degradation_score'] = ((df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)).clip(0.0, 1.0)

# Basic Physics
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked'].replace(0, 1)
    df['sat_efficiency'] = df['sat_efficiency'].clip(0, 5)
else:
    df['sat_efficiency'] = 0.0

# Rolling Features (конвертуємо в float32 одразу)
df = df.set_index('timestamp')
rolling_windows = ['5s', '10s']
cols_to_roll = ['cnoMean', 'sat_efficiency']

for col in cols_to_roll:
    if col in df.columns:
        for w in rolling_windows:
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

# ==========================================
# 3. ЧИСТКА ПАМ'ЯТІ
# ==========================================
print("🧹 Memory cleanup...")
cols_to_keep = features + ['degradation_score', 'timestamp']
df = df[cols_to_keep]

float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].astype(np.float32)
gc.collect()

# ==========================================
# 4. РОЗУМНЕ РОЗДІЛЕННЯ (SMART SPLIT)
# ==========================================
print("✂️ Splitting & Downsampling...")
split_date = pd.Timestamp('2025-12-01')

# ТЕСТОВИЙ набір залишаємо ПОВНИМ (щоб чесно перевірити)
test_df = df[df['timestamp'] >= split_date].copy()

# ТРЕНУВАЛЬНИЙ набір фільтруємо
train_full = df[df['timestamp'] < split_date]

# Стратегія: Залишаємо всі "Атаки" і тільки 25% "Спокою"
mask_attack = train_full['degradation_score'] > 0.05  # Всі підозрілі події
mask_safe = train_full['degradation_score'] <= 0.05   # Спокій

train_attack = train_full[mask_attack]
train_safe = train_full[mask_safe].sample(frac=0.25, random_state=42) # Беремо тільки чверть

# Об'єднуємо назад
train_df = pd.concat([train_attack, train_safe]).sample(frac=1, random_state=42) # Перемішуємо

print(f"   Original Train Size: {len(train_full)}")
print(f"   Optimized Train Size: {len(train_df)} (Memory Saved!)")

# Формуємо X та y
X_train = train_df[features]
y_train = train_df['degradation_score']
X_test = test_df[features]
y_test = test_df['degradation_score']

# Видаляємо зайве
del df, train_full, train_attack, train_safe, train_df, test_df
gc.collect()

# ==========================================
# 5. НАВЧАННЯ (Тільки Тюнінгована)
# ==========================================
print("\n🤖 Training Optimized Model...")

# Ми навчаємо тільки одну, найкращу модель, щоб економити ресурси
tuned_params = {
    'subsample': 0.7,
    'n_estimators': 100, 
    'max_depth': 6, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.7
}

model = XGBRegressor(
    objective='reg:logistic',
    n_jobs=-1,
    tree_method='hist', # Важливо!
    random_state=42,
    **tuned_params
)

model.fit(X_train, y_train)

print("✅ Model Trained!")

# Чистимо тренувальні дані перед тестом
del X_train, y_train
gc.collect()

# ==========================================
# 6. БЕЗПЕЧНЕ ПЕРЕДБАЧЕННЯ
# ==========================================
print("\n🔮 Predicting in batches...")

def predict_in_batches(model, X, batch_size=500000):
    num_samples = len(X)
    predictions = []
    print(f"   Processing {num_samples} samples...")
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_X = X.iloc[start:end]
        print(f"   Batch {start}-{end}...", end='\r')
        batch_pred = model.predict(batch_X)
        predictions.append(batch_pred)
        del batch_X, batch_pred
        gc.collect()
    print("\n   Done.")
    return np.concatenate(predictions)

y_pred = predict_in_batches(model, X_test)

# ==========================================
# 7. ОЦІНКА
# ==========================================
print("\n📊 Evaluation Results:")
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Feature Importance
plt.figure(figsize=(10, 8))
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(data=importance.head(15), x='Importance', y='Feature', palette='viridis')
plt.title('Top 15 Features (Final Model)')
plt.tight_layout()
plt.show()

# Zoom Plot
plt.figure(figsize=(12, 5))
subset = slice(10000, 10500)
plt.plot(y_test.iloc[subset].values, label='Actual', color='black', alpha=0.3)
plt.plot(y_pred[subset], label='Prediction', color='orange', linewidth=2)
plt.title('Prediction Sample')
plt.legend()
plt.show()