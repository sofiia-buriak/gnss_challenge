import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Завантаження даних
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)

# Перевірка формату часу
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Обчислення ковзного середнього (Rolling Mean)
# Використовуємо індексацію за часом для коректного розрахунку вікна '30s'
df_indexed = df.set_index('timestamp').sort_index()
df['vAcc_smooth'] = df_indexed['vAcc'].rolling('30s').mean().values

# 3. Візуалізація
plt.figure(figsize=(14, 8))

# Графік: Raw vs Smoothed (Log Scale)
plt.plot(df['timestamp'], df['vAcc'], label='Raw vAcc (Chattering)', color='lightgray', alpha=0.6, linewidth=0.8)
plt.plot(df['timestamp'], df['vAcc_smooth'], label='Smoothed vAcc (30s Rolling)', color='blue', linewidth=2)

plt.yscale('log') # Вмикаємо логарифмічну шкалу по Y
plt.title('Effect of Smoothing on Vertical Accuracy (Log Scale)')
plt.ylabel('Vertical Error (mm) [Log Scale]')
plt.xlabel('Timestamp')
plt.legend(loc='upper right')
plt.grid(True, which="both", ls="-", alpha=0.2)

# Додаємо лінію порогу (наприклад, 10 метрів = 10000 мм)
plt.axhline(y=10000, color='red', linestyle='--', label='Critical Threshold (10m)')

plt.tight_layout()
plt.show()

# 4. Аналіз Recall (Теоретичний)
# Припустимо, що атака - це коли vAcc > 10м.
threshold = 10000
true_attack = (df['vAcc_smooth'] > threshold) # Беремо згладжене як "істину" для прикладу стабільності

# Сирі спрацювання (з дірками)
raw_alerts = (df['vAcc'] > threshold)
# Згладжені спрацювання (суцільні)
smooth_alerts = (df['vAcc_smooth'] > threshold)

print(f"Кількість миттєвих 'провалів' (Raw < Threshold під час атаки): {(true_attack & ~raw_alerts).sum()}")
print(f"Це ті моменти, де ми втрачали Recall, а Smoothing їх виправив.")
