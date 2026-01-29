import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# 1. Завантаження (Тільки грудень, щоб було швидше)
DATA_PATH = 'data/processed/all_data_compressed.parquet'
print("📂 Loading data...")
df = pd.read_parquet(DATA_PATH)

# Перетворюємо час
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Вибір DOP-колонок
# Шукаємо все, що схоже на 'hdop', 'vdop', 'pdop', 'gdop'
dop_cols = [c for c in df.columns if 'dop' in c.lower()]
print(f"🔎 Found DOP columns: {dop_cols}")

if not dop_cols:
    print("❌ No DOP columns found in dataset!")
    exit()

# 3. Вибір шматка даних для візуалізації
# Беремо шматок десь із середини, щоб не дивитись на порожнечу
start_idx = len(df) // 2
window = 1000 # 1000 секунд
subset = df.iloc[start_idx : start_idx + window].copy()

# 4. ВІЗУАЛІЗАЦІЯ
plt.figure(figsize=(16, 10))

# --- ГРАФІК 1: Лінійна динаміка ---
plt.subplot(2, 1, 1)
for col in dop_cols:
    plt.plot(subset['timestamp'], subset[col], label=col, linewidth=2, alpha=0.8)

plt.title('DOP Metrics Dynamics (Linear Check)', fontsize=14)
plt.ylabel('DOP Value')
plt.xlabel('Time')
plt.legend()
plt.grid(True, alpha=0.3)

# --- ГРАФІК 2: Теплова карта кореляції ---
plt.subplot(2, 1, 2)
# Рахуємо кореляцію по ВСЬОМУ датасету (не тільки по шматочку), щоб було чесно
corr_matrix = df[dop_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt=".2f", linewidths=1)
plt.title('Correlation Matrix (Pearson Coefficient)', fontsize=14)

plt.tight_layout()
plt.savefig('dop_correlation_analysis.png', dpi=300)
print("✅ Graph saved as 'dop_correlation_analysis.png'")
plt.show()
