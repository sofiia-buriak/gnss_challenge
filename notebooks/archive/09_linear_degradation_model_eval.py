import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

# 1. Data Prep & Target Engineering
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

SAFE_LIMIT = 5.0  # meters
FAIL_LIMIT = 50.0  # meters

df['degradation_score'] = (df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)
df['degradation_score'] = df['degradation_score'].clip(0.0, 1.0)

# Feature engineering: sat_efficiency
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked']
    df['sat_efficiency'] = df['sat_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    df['sat_efficiency'] = 0
if 'cnoStd' not in df.columns:
    df['cnoStd'] = 0
for col in ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency']:
    df[col] = df[col].fillna(0)
# Lag features
for feat in ['sat_efficiency', 'cnoMean']:
    df[f'{feat}_lag1'] = df[feat].shift(1).fillna(0)

# Define input features (physics only, drop hAcc and overallPositionLabel)
forbidden = ['hAcc', 'vAcc', 'sAcc', 'pDOP', 'hDOP', 'vDOP', 'overallPositionLabel']
features = ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency', 'sat_efficiency_lag1', 'cnoMean_lag1']
features = [f for f in features if f in df.columns]

# 2. Training (Strict Split)
df_train = df[df['timestamp'] < pd.Timestamp('2025-12-01')]
df_test = df[df['timestamp'] >= pd.Timestamp('2025-12-01')]

X_train = df_train[features]
y_train = df_train['degradation_score']
X_test = df_test[features]
y_test = df_test['degradation_score']

# 3. Model Training
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:logistic',
    n_jobs=-1,
    tree_method='hist',
    eval_metric='rmse'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

df_test = df_test.copy()
df_test['predicted_score'] = y_pred

# 3A. Regression Metrics

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# 3B. Classification Proxy Metrics
if 'overallPositionLabel' in df_test.columns:
    y_true_bin = df_test['overallPositionLabel']
    roc_auc = roc_auc_score(y_true_bin, y_pred)
    print(f"ROC-AUC (Predicted Score vs. Unreliable): {roc_auc:.4f}")
else:
    print("overallPositionLabel not found in test set for ROC-AUC calculation.")

# 4. Visualization
sns.set_style("whitegrid")

# Boxplot by Class
if 'overallPositionLabel' in df_test.columns:
    plt.figure(figsize=(7,5))
    sns.boxplot(x='overallPositionLabel', y='predicted_score', data=df_test, palette=['#4CAF50', '#E74C3C'])
    plt.xticks([0,1],["Reliable (0)","Unreliable (1)"])
    plt.title('Predicted Degradation Score by Reliability Class')
    plt.ylabel('Predicted Degradation Score')
    plt.xlabel('overallPositionLabel')
    plt.tight_layout()
    plt.show()

# Residual Plot over Time
residuals = y_test - y_pred
plt.figure(figsize=(14,5))
plt.plot(df_test['timestamp'], residuals, color='purple', alpha=0.7)
plt.title('Residuals (Actual - Predicted) Over Time (Test Set)')
plt.xlabel('Timestamp')
plt.ylabel('Residual (y_test - y_pred)')
plt.tight_layout()
plt.show()
