import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Filter & Prepare
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df_anomalies = df[df['overallPositionLabel'] == 1].copy()

# Feature engineering: sat_efficiency
if 'numSV' in df_anomalies.columns and 'numSatsTracked' in df_anomalies.columns:
    df_anomalies['sat_efficiency'] = df_anomalies['numSV'] / df_anomalies['numSatsTracked']
    df_anomalies['sat_efficiency'] = df_anomalies['sat_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    df_anomalies['sat_efficiency'] = 0

# Select features for clustering
features = ['cnoMean', 'numSatsTracked', 'sat_efficiency', 'vAcc']
# Impute missing values
for col in features:
    if col in df_anomalies.columns:
        df_anomalies[col] = df_anomalies[col].fillna(0)
# Log transform vAcc to handle outliers
if 'vAcc' in df_anomalies.columns:
    df_anomalies['vAcc_log'] = np.log1p(df_anomalies['vAcc'])
else:
    df_anomalies['vAcc_log'] = 0

clustering_features = ['cnoMean', 'numSatsTracked', 'sat_efficiency', 'vAcc_log']
X = df_anomalies[clustering_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df_anomalies['attack_type_cluster'] = kmeans.fit_predict(X_scaled)

# Step 3: Cluster Identification (Automated Labeling)
cluster_stats = df_anomalies.groupby('attack_type_cluster').agg({'cnoMean':'mean', 'vAcc':'mean'})
# Identify clusters
jamming_cluster = cluster_stats['cnoMean'].idxmin()
spoofing_cluster = cluster_stats['cnoMean'].idxmax()
cluster_map = {jamming_cluster: 'Jamming', spoofing_cluster: 'Spoofing'}
df_anomalies['attack_type'] = df_anomalies['attack_type_cluster'].map(cluster_map)

# Step 4: Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_anomalies, x='cnoMean', y='sat_efficiency', hue='attack_type', alpha=0.5)
plt.title('Jamming vs Spoofing: cnoMean vs sat_efficiency')
plt.xlabel('cnoMean (Signal Strength)')
plt.ylabel('sat_efficiency')
plt.legend(title='Attack Type')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df_anomalies, x='attack_type', y='cnoMean')
plt.title('cnoMean Distribution by Attack Type')
plt.xlabel('Attack Type')
plt.ylabel('cnoMean (Signal Strength)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Output Stats
counts = df_anomalies['attack_type'].value_counts()
percentages = df_anomalies['attack_type'].value_counts(normalize=True) * 100
print('Attack Type Counts:')
print(counts)
print('\nAttack Type Percentages (%):')
print(percentages.round(2))