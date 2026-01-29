import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# 1. Load and sort data
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Target Engineering (Linear Ramp)
SAFE_LIMIT = 5000
FAIL_LIMIT = 50000

df['degradation_score'] = (df['hAcc'] - SAFE_LIMIT) / (FAIL_LIMIT - SAFE_LIMIT)
df['degradation_score'] = df['degradation_score'].clip(0.0, 1.0)

# 3. Feature Selection (Physics Only)
# Feature engineering: sat_efficiency
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked']
    df['sat_efficiency'] = df['sat_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    df['sat_efficiency'] = 0
for col in ['cnoMean', 'cnoStd', 'cnoMin', 'cnoMax', 'numSV', 'numSatsTracked', 'sat_efficiency']:
    df[col] = df[col].fillna(0)
# Lag features
for feat in ['sat_efficiency', 'cnoMean']:
    df[f'{feat}_lag1'] = df[feat].shift(1).fillna(0)

# Define input features
forbidden = ['hAcc', 'vAcc', 'sAcc', 'pDOP', 'hDOP', 'vDOP', 'overallPositionLabel']
features = ['cnoMean', 'cnoStd', 'cnoMin', 'cnoMax', 'numSV', 'numSatsTracked', 'sat_efficiency', 'sat_efficiency_lag1', 'cnoMean_lag1']
features = [f for f in features if f in df.columns]

# 4. Train/Test Split (time-series)
train_idx = df['timestamp'] < pd.Timestamp('2025-12-01')
test_idx = df['timestamp'] >= pd.Timestamp('2025-12-01')

X_train = df.loc[train_idx, features]
y_train = df.loc[train_idx, 'degradation_score']
X_test = df.loc[test_idx, features]
y_test = df.loc[test_idx, 'degradation_score']

# 5. Model Training (Regression)
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

# 6. Gray Zone Analysis
safe = (y_pred < 0.1).sum()
critical = (y_pred > 0.9).sum()
gray = ((y_pred >= 0.1) & (y_pred <= 0.9)).sum()
total = len(y_pred)
print(f"Safe (<0.1): {safe} ({100*safe/total:.2f}%)")
print(f"Critical (>0.9): {critical} ({100*critical/total:.2f}%)")
print(f"Gray Zone (0.1-0.9): {gray} ({100*gray/total:.2f}%)")

# 7. Visualization
plt.figure(figsize=(8,5))
sns.histplot(y_pred, bins=50, kde=True, color='orange', edgecolor='black')
plt.title('Distribution of Predicted Degradation Scores')
plt.xlabel('Predicted Degradation Score')
plt.ylabel('Count')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Transition Plot: Find a failure event
fail_idx = np.where((df.loc[test_idx, 'degradation_score'] > 0.9) & (df.loc[test_idx, 'degradation_score'].shift(1) <= 0.9))[0]
if len(fail_idx) > 0:
    idx = fail_idx[0]
    start = max(0, idx - 30)
    end = min(len(y_pred)-1, idx + 30)
    plt.figure(figsize=(12,5))
    plt.plot(range(start, end+1), y_pred[start:end+1], label='Predicted Score', color='orange')
    plt.plot(range(start, end+1), y_test.values[start:end+1], label='Actual Score', color='blue', alpha=0.7)
    plt.legend()
    plt.xlabel('Time (relative index)')
    plt.ylabel('Degradation Score')
    plt.title('Transition: Predicted vs Actual Degradation Score')
    plt.tight_layout()
    plt.show()
else:
    print('No failure event found in the test set for transition plot.')
