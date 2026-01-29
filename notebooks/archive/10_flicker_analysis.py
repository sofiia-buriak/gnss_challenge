import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import gc

# ==========================================
# 1. КОНФІГУРАЦІЯ ТА ФУНКЦІЇ
# ==========================================
DATA_PATH = 'data/processed/all_data_compressed.parquet'
TEST_START_DATE = '2025-12-01'

def prepare_physics_features(df):
    """Створює тільки фізичні фічі, ігноруючи координати."""
    # 1. Ефективність супутників
    if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
        df['sat_efficiency'] = df['numSV'] / df['numSatsTracked'].replace(0, 1)
        df['sat_efficiency'] = df['sat_efficiency'].clip(0, 5) # Прибираємо екстремальні викиди
    else:
        df['sat_efficiency'] = 0.0
    
    # 2. Лаги (історія сигналу) - важливо для трендів
    features_to_lag = ['cnoMean', 'sat_efficiency', 'numSV']
    for col in features_to_lag:
        if col in df.columns:
            # ВИПРАВЛЕННЯ: замість fillna(method='bfill') використовуємо bfill()
            df[f'{col}_lag1'] = df[col].shift(1).bfill()
            
    return df

def create_degradation_target(df):
    """Створює цільову змінну від 0.0 (добре) до 1.0 (погано)."""
    # Linear Ramp: 5м -> 0.0, 50м -> 1.0
    y = (df['hAcc'] - 5000.0) / (50000.0 - 5000.0)
    return y.clip(0.0, 1.0)

# ==========================================
# 2. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
print("📂 Loading data...")
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)

# Перетворення часу, якщо потрібно
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Генерація фіч для всього датасету (щоб лаги були коректні на стику дат)
df = prepare_physics_features(df)

# Список фіч для навчання (ТІЛЬКИ ФІЗИКА)
features = [
    'cnoMean', 'cnoMean_lag1',
    'sat_efficiency', 'sat_efficiency_lag1',
    'numSV', 'numSV_lag1',
    'numSatsTracked', 'cnoStd'
]
# Перевірка наявних колонок
features = [f for f in features if f in df.columns]

# ==========================================
# 3. РОЗДІЛЕННЯ TRAIN / TEST
# ==========================================
print("✂️ Splitting Train/Test...")
train_df = df[df['timestamp'] < TEST_START_DATE].copy()
test_df = df[df['timestamp'] >= TEST_START_DATE].copy()

# Звільняємо пам'ять
del df
gc.collect()

# ==========================================
# 4. НАВЧАННЯ МОДЕЛІ (REGRESSION)
# ==========================================
print(f"🤖 Training XGBRegressor on {len(train_df)} samples...")

X_train = train_df[features]
y_train = create_degradation_target(train_df)

model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:logistic', # Ідеально для виходу 0-1
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 5. ПЕРЕДБАЧЕННЯ ТА АНАЛІЗ (TEST)
# ==========================================
print(f"🔮 Predicting on Test set ({len(test_df)} samples)...")
X_test = test_df[features]
test_df['predicted_score'] = model.predict(X_test)

# --- АНАЛІЗ СТАБІЛЬНОСТІ (FLICKER ANALYSIS) ---
threshold = 0.5

# 1. Сирий сигнал (Raw)
test_df['raw_alert'] = (test_df['predicted_score'] > threshold).astype(int)
raw_flips = (test_df['raw_alert'].diff().abs() > 0).sum()

# 2. Згладжений сигнал (Smoothing)
# Rolling Mean: Беремо середнє за останні 5 секунд
window_sec = 5 
# Визначаємо розмір вікна в рядках (припускаємо 1 рядок = 1 сек, або обчислюємо медіану)
dt = test_df['timestamp'].diff().dt.total_seconds().median()
if np.isnan(dt) or dt == 0: dt = 1.0
window_rows = int(window_sec / dt)

print(f"   -> Applying Smoothing (Window: {window_sec}s, approx {window_rows} rows)...")

test_df['smoothed_score'] = test_df['predicted_score'].rolling(window=window_rows, min_periods=1).mean()
test_df['smoothed_alert'] = (test_df['smoothed_score'] > threshold).astype(int)
smooth_flips = (test_df['smoothed_alert'].diff().abs() > 0).sum()

# ==========================================
# 6. ВИВІД РЕЗУЛЬТАТІВ
# ==========================================
print("\n" + "="*40)
print("📊 FLICKER ANALYSIS RESULTS")
print("="*40)
print(f"Raw Flips (Chattering):      {raw_flips}")
print(f"Smoothed Flips (Stable):     {smooth_flips}")
reduction = (1 - smooth_flips/raw_flips) * 100
print(f"✅ Noise Reduction:          {reduction:.2f}%")
print("="*40)

# ==========================================
# 7. ВІЗУАЛІЗАЦІЯ (BARCODE PLOT)
# ==========================================
# Знаходимо цікавий момент переходу (де score зростає)
try:
    # Шукаємо індекс, де згладжений сигнал перемикається з 0 на 1
    transition_indices = np.where((test_df['smoothed_alert'].shift(1) == 0) & (test_df['smoothed_alert'] == 1))[0]
    
    if len(transition_indices) > 0:
        # Беремо перший чіткий перехід
        idx = transition_indices[0]
        # Вікно +/- 60 секунд
        subset_rows = 60
        start_pos = max(0, idx - subset_rows)
        end_pos = min(len(test_df), idx + subset_rows)
        
        subset = test_df.iloc[start_pos:end_pos]
        
        plt.figure(figsize=(14, 6))
        
        # Графік 1: Scores (Безперервні значення)
        plt.subplot(2, 1, 1)
        plt.plot(subset['timestamp'], subset['predicted_score'], color='lightgray', label='Raw Physics Score (Noisy)', alpha=0.7)
        plt.plot(subset['timestamp'], subset['smoothed_score'], color='orange', label='Smoothed Score (Stable)', linewidth=2)
        plt.axhline(threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
        plt.title('Raw vs Smoothed Degradation Score')
        plt.legend(loc='upper left')
        plt.ylabel('Score (0-1)')
        
        # Графік 2: Binary Alerts (Рішення Автопілота)
        plt.subplot(2, 1, 2)
        plt.step(subset['timestamp'], subset['raw_alert'], where='post', color='gray', linestyle=':', label='Raw Alert (Chattering)')
        plt.step(subset['timestamp'], subset['smoothed_alert'], where='post', color='green', linewidth=2, label='Smoothed Alert (Final Command)')
        plt.title('Autopilot Decision Signal (Barcode Plot)')
        plt.ylabel('Alert State (0/1)')
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
        print("📈 Plot generated successfully.")
    else:
        print("⚠️ No transitions found in test set to visualize.")

except Exception as e:
    print(f"⚠️ Error during plotting: {e}")

print("\n🚀 Pipeline completed successfully.")