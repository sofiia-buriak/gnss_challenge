import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# ==========================================
# 1. ПІДГОТОВКА (Швидка версія)
# ==========================================
print("📂 Loading data...")
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)

# Target Engineering
SAFE_LIMIT, FAIL_LIMIT = 5000, 50000
df['degradation_score'] = ((df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)).clip(0.0, 1.0)

# Feature Engineering (Full Model)
df['sat_efficiency'] = (df['numSV'] / df['numSatsTracked'].replace(0, 1)).clip(0, 5)
df = df.set_index('timestamp')
df['cnoMean_roll_mean'] = df['cnoMean'].rolling('10s').mean()
df['cnoMean_roll_std'] = df['cnoMean'].rolling('10s').std()
df = df.reset_index()

# Test Split (Грудень)
test_df = df[df['timestamp'].dt.month == 12].copy().reset_index(drop=True)
# Для економії часу тренуємо на маленькому шматку решти року, або завантажуємо готову модель
# Тут я швидко треную нову, щоб скрипт був автономним
train_df = df[df['timestamp'].dt.month < 12].sample(frac=0.2, random_state=42)

print("🤖 Training Full Model for Error Analysis...")
features = ['cnoMean', 'numSV', 'sat_efficiency', 'cnoMean_roll_mean', 'cnoMean_roll_std']
model = xgb.XGBRegressor(n_estimators=50, max_depth=6, objective='reg:logistic')
model.fit(train_df[features], train_df['degradation_score'])

# Predictions
test_df['pred'] = model.predict(test_df[features])
test_df['error'] = test_df['pred'] - test_df['degradation_score'] # + означає хибна тривога, - пропущена атака

# ==========================================
# 2. ПОШУК НАЙГІРШИХ ПОМИЛОК
# ==========================================

# Тип 1: False Positives (Паніка)
# Модель каже > 0.8, Реальність < 0.2
fp_cases = test_df[(test_df['pred'] > 0.8) & (test_df['degradation_score'] < 0.2)]
top_fp = fp_cases.sort_values('error', ascending=False).head(3)

# Тип 2: False Negatives (Сліпота - Найнебезпечніше!)
# Модель каже < 0.2, Реальність > 0.8
fn_cases = test_df[(test_df['pred'] < 0.2) & (test_df['degradation_score'] > 0.8)]
top_fn = fn_cases.sort_values('error', ascending=True).head(3)

print(f"\n🚩 Found {len(fp_cases)} False Positives (Panic)")
print(f"🚩 Found {len(fn_cases)} False Negatives (Blindness)")

# ==========================================
# 3. ВІЗУАЛІЗАЦІЯ "АНАТОМІЯ ПОМИЛКИ"
# ==========================================
def inspect_case(row_idx, case_type, case_num):
    # Беремо вікно +/- 30 секунд навколо помилки
    center_time = test_df.loc[row_idx, 'timestamp']
    start_time = center_time - pd.Timedelta(seconds=30)
    end_time = center_time + pd.Timedelta(seconds=30)
    
    window = test_df[(test_df['timestamp'] >= start_time) & (test_df['timestamp'] <= end_time)]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Графік 1: Прогноз vs Реальність
    axes[0].plot(window['timestamp'], window['degradation_score'], 'k-', label='Reality (Target)', linewidth=2)
    axes[0].plot(window['timestamp'], window['pred'], 'r--', label='Model Prediction', linewidth=2)
    axes[0].set_title(f"{case_type} Case #{case_num}: Prediction vs Truth", fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True)
    # Позначаємо момент помилки
    axes[0].axvline(center_time, color='orange', alpha=0.5)

    # Графік 2: Сила сигналу (CNO)
    axes[1].plot(window['timestamp'], window['cnoMean'], color='blue', label='Signal Strength (CNO)')
    axes[1].axhline(25, color='red', linestyle='--', label='Jamming Threshold')
    axes[1].set_ylabel('dBHz')
    axes[1].legend()
    axes[1].grid(True)
    
    # Графік 3: Супутники
    axes[2].plot(window['timestamp'], window['numSV'], color='green', label='Visible Satellites')
    axes[2].plot(window['timestamp'], window['numSatsTracked'], color='lime', linestyle=':', label='Tracked Satellites')
    axes[2].set_ylabel('Count')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'error_analysis_{case_type}_{case_num}.png', dpi=300)
    print(f"   Saved analysis to error_analysis_{case_type}_{case_num}.png")
    plt.show()

# Малюємо Топ-3 Хибні Тривоги
print("\n🔍 Analyzing False Positives (Why did the model panic?)...")
for i, (idx, row) in enumerate(top_fp.iterrows()):
    inspect_case(idx, "FalsePositive", i+1)

# Малюємо Топ-3 Пропуски
print("\n🔍 Analyzing False Negatives (Why did the model miss it?)...")
for i, (idx, row) in enumerate(top_fn.iterrows()):
    inspect_case(idx, "FalseNegative", i+1)
