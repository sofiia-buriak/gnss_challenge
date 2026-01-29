import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Завантаження даних
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)

# Переконуємось, що час у правильному форматі
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Обчислення ковзного середнього (30 секунд)
# Використовуємо індексацію за часом для коректного розрахунку вікна
df_indexed = df.set_index('timestamp').sort_index()
df['vAcc_smooth'] = df_indexed['vAcc'].rolling('30s').mean().values

# 3. Візуалізація
plt.figure(figsize=(14, 10))

# --- Графік 1: Часовий ряд (Log Scale) ---
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['vAcc'], label='Raw vAcc (mm)', color='lightgray', alpha=0.5, linewidth=0.8)
plt.plot(df['timestamp'], df['vAcc_smooth'], label='Smoothed vAcc (30s Moving Avg)', color='blue', linewidth=1.5)

plt.yscale('log') # Вмикаємо логарифмічну шкалу по Y
plt.title('Dynamics of Vertical Accuracy (vAcc) - Logarithmic Scale')
plt.ylabel('Vertical Error (mm) [Log Scale]')
plt.xlabel('Timestamp')
plt.legend(loc='upper right')
plt.grid(True, which="both", ls="-", alpha=0.2)

# --- Графік 2: Гістограма розподілу (Log X Scale) ---
plt.subplot(2, 1, 2)
# Фільтруємо нульові значення, щоб логарифм не зламався
vAcc_clean = df['vAcc'][df['vAcc'] > 0]
sns.histplot(vAcc_clean, bins=100, log_scale=True, color='purple', kde=True)
plt.title('Distribution of Vertical Accuracy (Log Scale)')
plt.xlabel('Vertical Error (mm) [Log Scale]')
plt.ylabel('Count')
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()

# 4. Статистика для "Сірої Зони" по висоті
# Визначимо пороги для vAcc (наприклад, 10м і 50м)
# 10 000 мм = 10 м
# 50 000 мм = 50 м
safe_v = (df['vAcc'] < 10000).sum()
gray_v = ((df['vAcc'] >= 10000) & (df['vAcc'] <= 50000)).sum()
crit_v = (df['vAcc'] > 50000).sum()
total = len(df)

print(f"\n📊 Vertical Accuracy Stats:")
print(f"Safe (<10m):      {safe_v} ({100*safe_v/total:.2f}%)")
print(f"Gray Zone (10-50m): {gray_v} ({100*gray_v/total:.2f}%)")
print(f"Critical (>50m):  {crit_v} ({100*crit_v/total:.2f}%)")
