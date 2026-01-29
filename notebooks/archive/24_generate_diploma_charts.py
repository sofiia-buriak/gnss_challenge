import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# НАЛАШТУВАННЯ
# ==========================================
OUTPUT_DIR = 'diploma_charts'
DATA_PATH = 'data/processed/all_data_compressed.parquet'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Стиль графіків (Науковий)
plt.style.use('seaborn-v0_8-whitegrid')
# Кольорова палітра
colors = ["#2c3e50", "#e74c3c", "#3498db", "#27ae60", "#f1c40f"]
sns.set_palette(sns.color_palette(colors))

# ==========================================
# 1. ЗАВАНТАЖЕННЯ (З оптимізацією)
# ==========================================
print("📂 Loading data...")

import gc # Garbage collector для очищення пам'яті

# 1. ВКАЖИ ТІЛЬКИ ПОТРІБНІ КОЛОНКИ
# Для дипломних графіків тобі точно не треба всі 100% колонок. 
# Вибери лише: час, цільову метрику (сигнал) і, наприклад, id супутника.
# Зміни назви у списку нижче на твої реальні:
REQUIRED_COLUMNS = [
    'timestamp',       # або 'time', 'datetime'
    'cnoMean',         # або 'signal_strength', 'cn0', 'raw_value'
    'numSV',           # або 'satellite_id', 'sv_id'
    'numSatsTracked',
    'hAcc',
    'pDOP', 'vDOP', 'hDOP' # додайте/змініть за потреби
]

print("⏳ Loading data with specific columns...")

try:
    # Завантажуємо тільки вибрані колонки — це економить до 80% RAM
    df = pd.read_parquet(
        DATA_PATH, 
        columns=REQUIRED_COLUMNS,
        engine='pyarrow'
    )
    # 2. ОПТИМІЗАЦІЯ ТИПІВ (Downcasting)
    # Перетворюємо float64 -> float32 (займає в 2 рази менше місця)
    fcols = df.select_dtypes('float').columns
    df[fcols] = df[fcols].astype('float32')
    icols = df.select_dtypes('integer').columns
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    print(f"✅ Data loaded! Shape: {df.shape}")
    print(f"🧠 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("Спробуй зменшити кількість колонок у списку REQUIRED_COLUMNS.")
    exit()

# Конвертація часу
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Для важких графіків беремо випадковий семпл (100к точок), щоб не чекати вічність
df_sample = df.sample(n=min(100000, len(df)), random_state=42)
print(f"✅ Data loaded: {len(df)} rows. Sampling 100k for heavy plots.")

# ==========================================
# 2. ГЕНЕРАЦІЯ ГРАФІКІВ
# ==========================================

# --- ГРУПА 1: СИГНАЛ (CNO) ---
print("📊 Generating Signal Charts...")

# 1. CNO Histogram (Детальний розподіл)
plt.figure(figsize=(10, 6))
sns.histplot(df['cnoMean'], bins=60, kde=True, color=colors[0], stat="percent")
plt.axvline(x=25, color=colors[1], linestyle='--', linewidth=2, label='Jamming Threshold (<25)')
plt.title('Distribution of Signal Strength (CNO)', fontsize=14)
plt.xlabel('Carrier-to-Noise Density (dBHz)')
plt.legend()
plt.savefig(f'{OUTPUT_DIR}/01_cno_distribution.png', dpi=300)
plt.close()

# 2. CNO Boxplot (Розкид значень)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df_sample['cnoMean'], color=colors[2])
plt.title('Signal Stability Analysis (Boxplot)', fontsize=14)
plt.xlabel('CNO (dBHz)')
plt.savefig(f'{OUTPUT_DIR}/02_cno_boxplot.png', dpi=300)
plt.close()

# 3. CNO Timeline (Динаміка) - беремо шматок де є провал
subset = df.iloc[5000:7000] # Довільний шматок або df_sample.sort_values...
plt.figure(figsize=(12, 5))
plt.plot(subset['timestamp'], subset['cnoMean'], color=colors[0], linewidth=1)
plt.title('Signal Drop Detection (Timeline Fragment)', fontsize=14)
plt.ylabel('CNO (dBHz)')
plt.savefig(f'{OUTPUT_DIR}/03_cno_timeline.png', dpi=300)
plt.close()


# --- ГРУПА 2: СУПУТНИКИ (Satellites) ---
print("🛰️ Generating Satellite Charts...")

# 4. Satellites Count (Bar Plot)
plt.figure(figsize=(10, 6))
sns.countplot(x=df_sample['numSV'], color=colors[2])
plt.title('Visible Satellites Count Distribution', fontsize=14)
plt.xlabel('Number of Visible Satellites')
plt.savefig(f'{OUTPUT_DIR}/04_satellites_count.png', dpi=300)
plt.close()

# 5. Tracked vs Visible (Scatter Density)
if 'numSatsTracked' in df.columns:
    plt.figure(figsize=(8, 8))
    plt.hist2d(df_sample['numSV'], df_sample['numSatsTracked'], bins=30, cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.plot([0, 40], [0, 40], 'r--', label='Ideal 1:1')
    plt.title('Efficiency: Visible vs Tracked Satellites', fontsize=14)
    plt.xlabel('Visible (numSV)')
    plt.ylabel('Tracked (Used)')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}/05_sat_efficiency.png', dpi=300)
    plt.close()


# --- ГРУПА 3: ГЕОМЕТРІЯ (DOP) ---
print("📐 Generating Geometry Charts...")

# 6. DOP Correlation Heatmap
dop_cols = [c for c in df.columns if 'dop' in c.lower()]
if dop_cols:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[dop_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix: Geometric Factors', fontsize=14)
    plt.savefig(f'{OUTPUT_DIR}/06_dop_heatmap.png', dpi=300)
    plt.close()

# 7. PDOP Distribution (Violin Plot - красиво показує щільність)
if 'pDOP' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df_sample['pDOP'], color=colors[3])
    plt.title('PDOP Density Distribution', fontsize=14)
    plt.xlabel('Position Dilution of Precision')
    plt.savefig(f'{OUTPUT_DIR}/07_pdop_violin.png', dpi=300)
    plt.close()


# --- ГРУПА 4: ЦІЛЬОВА ЗМІННА (ACCURACY) ---
print("🎯 Generating Accuracy Charts...")

# 8. hAcc Histogram (Log Scale - ОБОВ'ЯЗКОВО)
plt.figure(figsize=(10, 6))
sns.histplot(df['hAcc'], bins=100, log_scale=True, color=colors[1])
plt.axvline(x=5000, color='green', linestyle='--', label='Normal (<5m)')
plt.axvline(x=50000, color='red', linestyle='--', label='Critical (>50m)')
plt.title('Horizontal Accuracy Error Distribution (Log Scale)', fontsize=14)
plt.xlabel('Error (mm)')
plt.legend()
plt.savefig(f'{OUTPUT_DIR}/08_hacc_log_hist.png', dpi=300)
plt.close()

# 9. The "Hockey Stick" (CNO vs hAcc) - ДОКАЗ ФІЗИКИ
plt.figure(figsize=(10, 6))
plt.scatter(df_sample['cnoMean'], df_sample['hAcc'], alpha=0.2, s=15, color='#8e44ad')
plt.yscale('log')
plt.title('Correlation: Signal Strength vs Accuracy', fontsize=14)
plt.xlabel('Signal Strength (CNO)')
plt.ylabel('Accuracy Error (mm) [Log Scale]')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig(f'{OUTPUT_DIR}/09_physics_proof.png', dpi=300)
plt.close()

# 10. Pairplot (Швидкий огляд всього з усім) - Тільки для семплу
cols_to_plot = ['cnoMean', 'numSV', 'hAcc']
if 'pDOP' in df.columns: cols_to_plot.append('pDOP')

print("   Generating final complex pairplot (might take a moment)...")
sns.pairplot(df_sample[cols_to_plot], diag_kind='kde', plot_kws={'alpha': 0.1, 's': 5})
plt.savefig(f'{OUTPUT_DIR}/10_global_pairplot.png', dpi=300)
plt.close()

print(f"\n✅ ГОТОВО! Всі 10 графіків збережено в папку: {os.path.abspath(OUTPUT_DIR)}")
