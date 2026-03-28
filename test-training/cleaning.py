import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json

print("1. Loading Simulated Data...")
# Load the perfectly clean data we extracted from SQLite
df = pd.read_csv("ai_ready_crowd_data.csv")

# 2. Separate Features (X) and Target (y)
y = df['current_crowd']
X = df.drop(columns=['current_crowd'])

# 3. Split the Data (80% for training, 20% for testing)
# Since the simulated data is already chronological, we don't need to sort it
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Training on the first {len(X_train)} time-steps...")
print(f"Testing the AI on the final {len(X_test)} time-steps...\n")

# --- TRAIN THE AI MODEL ---
print("2. Training XGBoost Model on Simulated Data...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
print("Training complete!\n")

# --- TEST THE AI ---
predictions = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("--- MODEL EVALUATION ---")
print(f"Mean Absolute Error: {mae:.2f} people")
print("This means on average, the AI is off by this many people per simulated hour.\n")

print("--- REAL EXAMPLES ---")
for i in range(3):
    actual = y_test.iloc[i]
    predicted = predictions[i]
    print(f"Test Step {i+1} | Actual Crowd: {actual} | AI Predicted: {predicted:.0f}")

# --- SAVING MODEL FOR PRODUCTION ---
print("\n--- SAVING MODEL FOR PRODUCTION ---")
model_filename = "xgboost_simulated_model.joblib"
joblib.dump(xgb_model, model_filename)
print(f" Model successfully saved as: {model_filename}")

columns_filename = "simulated_model_columns.json"
with open(columns_filename, 'w') as f:
    json.dump(list(X_train.columns), f)
print(f" Column structure saved as: {columns_filename}")

print("\n Your AI has successfully learned the patterns of your simulated venue!")