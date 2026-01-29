import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preprocessing & Feature Engineering
# Load data
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)

# Impute missing values
dop_cols = ['pDOP', 'hDOP']
signal_cols = ['cnoMean']
for col in dop_cols:
    if col in df.columns:
        df[col] = df[col].fillna(-1)
for col in signal_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Feature engineering: sat_efficiency
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked']
    df['sat_efficiency'] = df['sat_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    df['sat_efficiency'] = 0

# Lag features (sorted by time)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
df['cnoMean_lag1'] = df['cnoMean'].shift(1).fillna(0)
df['sat_efficiency_lag1'] = df['sat_efficiency'].shift(1).fillna(0)

# 2. Define Feature Sets
# Set A: Baseline (all features except labels)
label_cols = ['timestamp', 'overallPositionLabel', 'horizontalPositionLabel', 'verticalPositionLabel']
features_A = [col for col in df.columns if col not in label_cols]

# Set B: Physics Only (drop accuracy and DOP columns)
giveaway_cols = ['hAcc', 'vAcc', 'sAcc', 'pDOP', 'hDOP', 'vDOP', 'nDOP', 'eDOP']
features_B = [col for col in features_A if col not in giveaway_cols]

# Remove any datetime/object columns from features
for drop_type in ['datetime64[ns]', 'datetime64', 'timedelta64[ns]', 'object']:
    features_A = [col for col in features_A if not np.issubdtype(df[col].dtype, np.datetime64) and df[col].dtype != 'O']
    features_B = [col for col in features_B if not np.issubdtype(df[col].dtype, np.datetime64) and df[col].dtype != 'O']

# 3. Training (Strict Time-Series Split)
train_idx = df['timestamp'] < pd.Timestamp('2025-12-01')
test_idx = df['timestamp'] >= pd.Timestamp('2025-12-01')

# Sample up to 500,000 rows for train and test to avoid memory error
TRAIN_SAMPLE = 500_000
TEST_SAMPLE = 500_000

X_train_A = df.loc[train_idx, features_A].sample(n=min(TRAIN_SAMPLE, train_idx.sum()), random_state=42)
y_train = df.loc[X_train_A.index, 'overallPositionLabel']
X_test_A = df.loc[test_idx, features_A].sample(n=min(TEST_SAMPLE, test_idx.sum()), random_state=42)
y_test = df.loc[X_test_A.index, 'overallPositionLabel']

X_train_B = df.loc[X_train_A.index, features_B]
X_test_B = df.loc[X_test_A.index, features_B]

# Handle class imbalance
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# Model A: Baseline
model_A = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)
model_A.fit(X_train_A, y_train)
y_pred_A = model_A.predict(X_test_A)

# Model B: Physics Only
model_B = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)
model_B.fit(X_train_B, y_train)
y_pred_B = model_B.predict(X_test_B)

# 4. Evaluation & Comparison
print('--- Classification Report: Model A (Baseline) ---')
print(classification_report(y_test, y_pred_A, digits=3))
mcc_A = matthews_corrcoef(y_test, y_pred_A)

print('\n--- Classification Report: Model B (Physics Only) ---')
print(classification_report(y_test, y_pred_B, digits=3))
mcc_B = matthews_corrcoef(y_test, y_pred_B)

# F1-score drop analysis
from sklearn.metrics import f1_score
f1_A = f1_score(y_test, y_pred_A)
f1_B = f1_score(y_test, y_pred_B)
print(f"\nF1-score drop: Baseline={f1_A:.3f} -> Physics Only={f1_B:.3f} | Drop={f1_A-f1_B:.3f}")

# Feature Importance for Model B
importances_B = model_B.feature_importances_
feat_imp_B = pd.Series(importances_B, index=X_train_B.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feat_imp_B.head(10).plot(kind='barh', color='darkorange')
plt.gca().invert_yaxis()
plt.title('Top 10 Feature Importances (Physics Only Model)')
plt.xlabel('Importance')
plt.show()

print('Top 10 features (Physics Only Model):')
print(feat_imp_B.head(10))