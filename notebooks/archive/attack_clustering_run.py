import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
print("📂 Loading data...")
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)

# Розрахунок target (щоб знайти де була атака)
SAFE_LIMIT, FAIL_LIMIT = 5000, 50000
df['degradation_score'] = ((df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)).clip(0.0, 1.0)

# Додаткові фічі
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = (df['numSV'] / df['numSatsTracked'].replace(0, 1)).clip(0, 5)
else:
    df['sat_efficiency'] = 0

# ==========================================
# 2. ФІЛЬТРАЦІЯ АНОМАЛІЙ
# ==========================================
print("🔍 Filtering anomalies (Attacks)...")

# Ми кластеризуємо ТІЛЬКИ ті моменти, де була проблема (score > 0.5)
attacks = df[df['degradation_score'] > 0.5].copy()

# Видаляємо NaN у колонках для кластеризації
features_to_cluster = ['cnoMean', 'sat_efficiency']
attacks = attacks.dropna(subset=features_to_cluster)

if len(attacks) < 100:
    print("⚠️ Too few attack samples found for clustering!")
    exit()

print(f"   Found {len(attacks)} anomaly samples to analyze.")

# ==========================================
# 3. K-MEANS CLUSTERING
# ==========================================
# Вибираємо ознаки для кластеризації:
# 1. cnoMean (Сила сигналу) - головний розрізнювач
# 2. sat_efficiency (Геометрія) - як поводяться супутники
features_to_cluster = ['cnoMean', 'sat_efficiency']

# Нормалізація (StandardScaler) - обов'язково для K-Means!
scaler = StandardScaler()
X = attacks[features_to_cluster]
X_scaled = scaler.fit_transform(X)

# Запускаємо K-Means на 2 кластери (Jamming vs Spoofing)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
attacks['cluster'] = kmeans.fit_predict(X_scaled)

# ==========================================
# 4. АВТОМАТИЧНЕ ВИЗНАЧЕННЯ ТИПІВ
# ==========================================
# Дивимось на середній CNO у кожному кластері
cluster_centers = attacks.groupby('cluster')['cnoMean'].mean()
print("\n📊 Cluster Centers (Average CNO):")
print(cluster_centers)

# Логіка: Там, де CNO менший - це Jamming. Де більший - Spoofing/Multipath.
jamming_cluster_id = cluster_centers.idxmin()
spoofing_cluster_id = cluster_centers.idxmax()

attacks['Attack Type'] = attacks['cluster'].map({
    jamming_cluster_id: 'Jamming (Low Signal)',
    spoofing_cluster_id: 'Spoofing/Interference (High Signal)'
})

print(f"   ✅ Identified Cluster {jamming_cluster_id} as Jamming")
print(f"   ✅ Identified Cluster {spoofing_cluster_id} as Spoofing")

# ==========================================
# 5. ВІЗУАЛІЗАЦІЯ
# ==========================================
plt.figure(figsize=(10, 6))

# Малюємо точки
sns.scatterplot(
    data=attacks, 
    x='cnoMean', 
    y='sat_efficiency', 
    hue='Attack Type', 
    palette={'Jamming (Low Signal)': '#e74c3c', 'Spoofing/Interference (High Signal)': '#3498db'},
    alpha=0.6,
    s=15
)

plt.title('Unsupervised Classification of GNSS Attack Types', fontsize=14)
plt.xlabel('Signal Strength (cnoMean)')
plt.ylabel('Satellite Efficiency (Visible / Tracked)')
plt.axvline(x=25, color='gray', linestyle='--', label='Typical Jamming Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('attack_clusters.png', dpi=300)
print("\n✅ Plot saved to 'attack_clusters.png'")
plt.show()
