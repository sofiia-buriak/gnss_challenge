import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and sort data
DATA_PATH = 'data/processed/all_data_compressed.parquet'
df = pd.read_parquet(DATA_PATH)
df = df.sort_values('timestamp').reset_index(drop=True)
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Target Engineering: Forecast 5 seconds ahead
df['y_future'] = df['overallPositionLabel'].shift(-5)
# Drop last 5 rows (future target is NaN)
df = df.iloc[:-5].copy()

# 3. Feature Engineering (Physics Only)
# Impute signal columns
for col in ['cnoMean', 'numSatsTracked']:
    if col in df.columns:
        df[col] = df[col].fillna(0)
if 'numSV' in df.columns:
    df['numSV'] = df['numSV'].fillna(0)
# sat_efficiency
if 'numSV' in df.columns and 'numSatsTracked' in df.columns:
    df['sat_efficiency'] = df['numSV'] / df['numSatsTracked']
    df['sat_efficiency'] = df['sat_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
else:
    df['sat_efficiency'] = 0
# cnoStd
if 'cnoStd' not in df.columns:
    df['cnoStd'] = 0

# Lag features
for feat in ['cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency']:
    df[f'{feat}_lag1'] = df[feat].shift(1).fillna(0)
    df[f'{feat}_lag3'] = df[feat].shift(3).fillna(0)

# Select features (Physics Only + lags)
features = [
    'cnoMean', 'cnoStd', 'numSV', 'numSatsTracked', 'sat_efficiency',
    'cnoMean_lag1', 'cnoMean_lag3',
    'cnoStd_lag1', 'cnoStd_lag3',
    'numSV_lag1', 'numSV_lag3',
    'numSatsTracked_lag1', 'numSatsTracked_lag3',
    'sat_efficiency_lag1', 'sat_efficiency_lag3'
]

# 4. Train/Test Split (strict time-based)
train_idx = df['timestamp'] < pd.Timestamp('2025-12-01')
test_idx = df['timestamp'] >= pd.Timestamp('2025-12-01')

# To avoid memory issues, sample up to 500,000 rows for each set
TRAIN_SAMPLE = 500_000
TEST_SAMPLE = 500_000
X_train = df.loc[train_idx, features].sample(n=min(TRAIN_SAMPLE, train_idx.sum()), random_state=42)
y_train = df.loc[X_train.index, 'y_future']
X_test = df.loc[test_idx, features].sample(n=min(TEST_SAMPLE, test_idx.sum()), random_state=42)
y_test = df.loc[X_test.index, 'y_future']

# 5. Model Training
neg, pos = np.bincount(y_train.astype(int))
scale_pos_weight = neg / pos
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 6. Evaluation
print('--- Classification Report: Forecasting Model (t+5s) ---')
print(classification_report(y_test, y_pred, digits=3))
f1_forecast = f1_score(y_test, y_pred)
mcc_forecast = matthews_corrcoef(y_test, y_pred)
print(f"F1-score (Forecasting Model): {f1_forecast:.3f}")
print(f"Matthews Correlation Coefficient: {mcc_forecast:.3f}")

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 5))
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Reliable (0)', 'Unreliable (1)'], 
            yticklabels=['Reliable (0)', 'Unreliable (1)'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Forecasting Model, Normalized)')
plt.show()

# 8. "Crystal Ball" Plot: 2-minute window around a failure event
# Find a window with at least one failure in y_test
test_df = df.loc[X_test.index].copy().reset_index(drop=True)
test_df['y_pred'] = y_pred
test_df['proba'] = y_proba
# Find an iloc where a failure starts
failure_indices = test_df.index[test_df['y_future'].diff().fillna(0) == 1].tolist()
if failure_indices:
    idx = failure_indices[0]
    start = max(0, idx - 60)
    end = min(len(test_df) - 1, idx + 60)
    window = test_df.iloc[start:end+1].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(window['timestamp'], window['proba'], label='Predicted Failure Probability', color='orange')
    plt.plot(window['timestamp'], window['y_future'], label='Actual Future Failure', color='red', alpha=0.5)
    plt.legend()
    plt.xlabel('Timestamp')
    plt.ylabel('Probability / Label')
    plt.title('"Crystal Ball" Plot: Early Warning Prediction vs. Actual Failure')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print('No failure event found in the sampled test set for Crystal Ball plot.')

# 9. (Optional) Compare with Detection Model F1 if available
# (User should provide previous F1 for direct comparison)
