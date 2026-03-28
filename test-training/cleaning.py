import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
import json
import joblib

# --- LOAD DATA ---
df = pd.read_csv('ai_ready_crowd_data.csv')

# --- CLEANING ---
df = df.rename(columns={
    'Crowd_Count (in Thousands)': 'Crowd_Count',
    'Temperature (C)': 'Temperature'
})

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

columns_to_drop = ['Unnamed: 12', 'Special_Features']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# --- SORT ---
df = df.sort_values('Date')

# =====================================================
# 🔥 FEATURE ENGINEERING (BIG IMPROVEMENT)
# =====================================================

# Time features
df["hour"] = pd.to_datetime(df["Time"], errors='coerce').dt.hour
df["month"] = df["Date"].dt.month
df["is_weekend"] = df["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)

# Cyclic encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Lag features
df["lag_1"] = df["Crowd_Count"].shift(1)
df["lag_7"] = df["Crowd_Count"].shift(7)

# Rolling averages
df["rolling_mean_3"] = df["Crowd_Count"].rolling(3).mean()
df["rolling_mean_7"] = df["Crowd_Count"].rolling(7).mean()

# Drop NaNs created by lag/rolling
df = df.dropna()

# =====================================================
# 🔥 ENCODING
# =====================================================
categorical_cols = ['Place', 'Weather', 'Day_of_Week', 'Holiday', 'Event', 'Region', 'Transportation_Type']
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])

# =====================================================
# SPLIT
# =====================================================
y = np.log1p(df['Crowd_Count'])
X = df.drop(columns=['Crowd_Count', 'Date', 'Time'])

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

split_index = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# =====================================================
# 🔥 HYPERPARAMETER TUNING
# =====================================================
params = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

model = XGBRegressor(random_state=42)

search = RandomizedSearchCV(
    model,
    params,
    n_iter=10,
    cv=TimeSeriesSplit(n_splits=3),
    verbose=1
)

print("Tuning model...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

# =====================================================
# 🔥 TRAIN FINAL MODEL
# =====================================================
print("Training final model...")
best_model.fit(X_train, y_train)

# =====================================================
# 🔥 EVALUATION
# =====================================================
preds = best_model.predict(X_test)
preds = np.expm1(preds)
y_test_actual = np.expm1(y_test)

mae = mean_absolute_error(y_test_actual, preds)
mape = mean_absolute_percentage_error(y_test_actual, preds)

print(f"\nMAE: {mae:.2f}")
print(f"MAPE: {mape * 100:.2f}%")

print("\n--- REAL EXAMPLES ---")
for i in range(3):
    print(f"Actual: {y_test.iloc[i]} | Predicted: {preds[i]:.1f}")
    
print("\n--- SAVING MODEL FOR PRODUCTION ---")

# 1. Save the trained XGBoost model to a file
model_filename = "xgboost_crowd_model.joblib"
joblib.dump(best_model, model_filename)
print(f" Model successfully saved as: {model_filename}")

# 2. CRITICAL: Save the exact column names!
# Because we used 'get_dummies' earlier, the AI expects exact column names like 'Weather_Sunny'
# We must save this list so our Node.js server knows exactly how to format incoming data.
columns_filename = "model_columns.json"
with open(columns_filename, 'w') as f:
    json.dump(list(X_train.columns), f)
print(f" Column structure saved as: {columns_filename}")

print("\n Your AI is officially ready to be plugged into your web app!")