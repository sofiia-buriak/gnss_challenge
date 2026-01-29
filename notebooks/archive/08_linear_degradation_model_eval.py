import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score

# 1. Load Data (Assume same as in model script)
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Target engineering (must match model script)
SAFE_LIMIT = 5000
FAIL_LIMIT = 50000
df['degradation_score'] = (df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)
df['degradation_score'] = df['degradation_score'].clip(0.0, 1.0)

# Feature engineering (must match model script)
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked']
    df['sat_efficiency'] = df['sat_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    df['sat_efficiency'] = 0
if 'cnoStd' not in df.columns:
    df['cnoStd'] = 0
for col in ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency']:
    df[col] = df[col].fillna(0)
for feat in ['sat_efficiency', 'cnoMean']:
    df[f'{feat}_lag1'] = df[feat].shift(1).fillna(0)
features = ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency', 'sat_efficiency_lag1', 'cnoMean_lag1']
features = [f for f in features if f in df.columns]

# Time-based split (must match model script)
train_idx = df['timestamp'] < pd.Timestamp('2025-12-01')
test_idx = df['timestamp'] >= pd.Timestamp('2025-12-01')
df_test = df.loc[test_idx].copy()
X_test = df_test[features]
y_test = df_test['degradation_score']

# Load model and predictions (re-run model for reproducibility)
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:logistic',
    n_jobs=-1,
    tree_method='hist',
    eval_metric='rmse'
)
# Re-train on train set for reproducibility
X_train = df.loc[train_idx, features]
y_train = df.loc[train_idx, 'degradation_score']
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
df_test['y_pred'] = y_pred

# 1. Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

# 2. Classification Proxy Metrics
# Use overallPositionLabel as binary target (0: Reliable, 1: Unreliable)
if 'overallPositionLabel' in df_test.columns:
    y_true_bin = df_test['overallPositionLabel']
    roc_auc = roc_auc_score(y_true_bin, y_pred)
    pr_auc = average_precision_score(y_true_bin, y_pred)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
else:
    print("overallPositionLabel not found in test set.")

# 3. Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(12,4))
plt.plot(df_test['timestamp'], residuals, color='purple', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residuals (y_test - y_pred) Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Residual')
plt.tight_layout()
plt.show()

# 4. Separation Analysis (Boxplot)
if 'overallPositionLabel' in df_test.columns:
    plt.figure(figsize=(7,5))
    sns.boxplot(x='overallPositionLabel', y='y_pred', data=df_test, palette=['green','red'])
    plt.xticks([0,1],["Reliable (0)","Unreliable (1)"])
    plt.title('Predicted Degradation Score by Reliability Class')
    plt.ylabel('Predicted Degradation Score')
    plt.xlabel('overallPositionLabel')
    plt.tight_layout()
    plt.show()
else:
    print("overallPositionLabel not found for boxplot.")

# 5. Density Plot (KDE Separation)
if 'overallPositionLabel' in df_test.columns:
    plt.figure(figsize=(8,5))
    sns.kdeplot(df_test.loc[df_test['overallPositionLabel']==0, 'y_pred'], label='Reliable (0)', color='green', fill=True, alpha=0.5)
    sns.kdeplot(df_test.loc[df_test['overallPositionLabel']==1, 'y_pred'], label='Unreliable (1)', color='red', fill=True, alpha=0.5)
    plt.title('KDE of Predicted Score by Reliability Class')
    plt.xlabel('Predicted Degradation Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("overallPositionLabel not found for KDE plot.")
