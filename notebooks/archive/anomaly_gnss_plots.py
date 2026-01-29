import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

df = pd.read_parquet('data/processed/all_data_compressed.parquet')
os.makedirs('anomaly_investigation', exist_ok=True)

# 1. "Парадокс сильного сигналу" (оптимізовано для великих даних)
plt.figure(figsize=(8,6))
mask_anomaly = (df['cnoMean'] > 35) & (df['hAcc'] > 50000)
# Вибірка до 100 000 звичайних точок
normal_idx = df[~mask_anomaly].sample(n=min(100_000, (~mask_anomaly).sum()), random_state=42).index
plt.scatter(df.loc[normal_idx, 'cnoMean'], df.loc[normal_idx, 'hAcc'], c='grey', alpha=0.3, label='Звичайні точки', s=2)
plt.scatter(df.loc[mask_anomaly, 'cnoMean'], df.loc[mask_anomaly, 'hAcc'],
            c='red', alpha=0.9, label='High Signal / High Error', s=10)
plt.yscale('log')
plt.xlabel('cnoMean (dBHz)')
plt.ylabel('hAcc (мм, log scale)')
plt.title('Парадокс сильного сигналу')
plt.legend(loc='upper right')
if mask_anomaly.sum() > 0:
    for idx in df[mask_anomaly].index[:3]:
        plt.annotate('Аномалія!',
                     (df.loc[idx, 'cnoMean'], df.loc[idx, 'hAcc']),
                     textcoords="offset points", xytext=(10,10), ha='left',
                     arrowprops=dict(arrowstyle="->", color='red'))
plt.tight_layout()
plt.savefig('anomaly_investigation/high_signal_high_error.png', dpi=300)
plt.close()

# 2. "Привидні супутники" (оптимізовано для великих даних)
plt.figure(figsize=(7,7))
jitter = lambda x: x + np.random.uniform(-0.15, 0.15, size=len(x))
mask_ghost = df['numSatsTracked'] > df['numSV']
# Вибірка до 100 000 звичайних точок
normal_idx = df[~mask_ghost].sample(n=min(100_000, (~mask_ghost).sum()), random_state=42).index
x = jitter(df.loc[normal_idx, 'numSV'])
y = jitter(df.loc[normal_idx, 'numSatsTracked'])
x_ghost = jitter(df.loc[mask_ghost, 'numSV'])
y_ghost = jitter(df.loc[mask_ghost, 'numSatsTracked'])
plt.scatter(x, y, c='grey', alpha=0.3, label='Звичайні точки', s=2)
plt.scatter(x_ghost, y_ghost, c='red', alpha=0.8, label='Tracked > Visible', s=10)
plt.plot([df['numSV'].min()-1, df['numSV'].max()+1],
         [df['numSV'].min()-1, df['numSV'].max()+1], 'k--', lw=2, label='x = y')
plt.xlabel('numSV (Видимі супутники)')
plt.ylabel('numSatsTracked (Використані супутники)')
plt.title('Привидні супутники')
plt.legend(loc='upper left')
if mask_ghost.sum() > 0:
    plt.annotate('Аномалія: більше використано,\nніж видно!',
                 (x_ghost.iloc[0], y_ghost.iloc[0]), xytext=(30,10), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='red'))
plt.tight_layout()
plt.savefig('anomaly_investigation/ghost_satellites.png', dpi=300)
plt.close()

# 3. "Розрив вертикальної геометрії" (оптимізовано для великих даних)
plt.figure(figsize=(8,6))
mask_vgap = (df['vDOP'] > 5.0) & (df['hDOP'] < 2.0)
# Вибірка до 100 000 звичайних точок
normal_idx = df[~mask_vgap].sample(n=min(100_000, (~mask_vgap).sum()), random_state=42).index
plt.scatter(df.loc[normal_idx, 'hDOP'], df.loc[normal_idx, 'vDOP'], c='grey', alpha=0.3, label='Звичайні точки', s=2)
plt.scatter(df.loc[mask_vgap, 'hDOP'], df.loc[mask_vgap, 'vDOP'],
            c='orange', alpha=0.9, label='vDOP > 5.0 & hDOP < 2.0', s=10)
plt.xlabel('hDOP')
plt.ylabel('vDOP')
plt.title('Розрив вертикальної геометрії')
plt.legend(loc='upper left')
if mask_vgap.sum() > 0:
    idx = df[mask_vgap].index[0]
    plt.annotate('Дисбаланс!\n"Колодязь"',
                 (df.loc[idx, 'hDOP'], df.loc[idx, 'vDOP']),
                 xytext=(20, -30), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='orange'))
plt.tight_layout()
plt.savefig('anomaly_investigation/vertical_gap.png', dpi=300)
plt.close()

# 4. "Мертві нулі"
plt.figure(figsize=(8,5))
cno_zeros = df['cnoMean'][(df['cnoMean'] >= 0) & (df['cnoMean'] <= 10)]
bins = np.arange(-0.5, 10.6, 1)
n, bins, patches = plt.hist(cno_zeros, bins=bins, color='grey', edgecolor='black', alpha=0.7)
for patch, left in zip(patches, bins[:-1]):
    if left < 0.5 and left >= -0.5:
        patch.set_facecolor('red')
plt.xlabel('cnoMean (dBHz)')
plt.ylabel('Кількість')
plt.title('Мертві нулі: розподіл cnoMean (0..10 dBHz)')
plt.annotate('Мертві нулі', xy=(0, n[0]), xytext=(2, n[0]+max(n)*0.1),
             arrowprops=dict(arrowstyle="->", color='red'), color='red')
plt.tight_layout()
plt.savefig('anomaly_investigation/dead_zeros_cnoMean.png', dpi=300)
plt.close()

# 5. "Дірки в часі"
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
dt = df['timestamp'].diff().dt.total_seconds()
mask_gap = dt > 1.5
plt.figure(figsize=(12,5))
plt.scatter(df['timestamp'], dt, c='grey', alpha=0.3, label='Звичайні інтервали')
plt.scatter(df['timestamp'][mask_gap], dt[mask_gap], c='red', alpha=0.9, label='Gap > 1.5s')
plt.xlabel('Час')
plt.ylabel('Інтервал між записами (сек)')
plt.title('Дірки в часі')
plt.legend()
if mask_gap.sum() > 0:
    idx = df.index[mask_gap][0]
    plt.annotate('Розрив > 1.5s',
                 (df.loc[idx, 'timestamp'], dt.loc[idx]),
                 xytext=(30, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='red'))
plt.tight_layout()
plt.savefig('anomaly_investigation/time_gaps.png', dpi=300)
plt.close()
