import torch
import pandas as pd
from chronos import Chronos2Pipeline
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "test-training/chronos_covariate_data.csv"
PREDICTION_LENGTH = 10
MAX_CAPACITY = 50
SURGE_THRESHOLD = 15

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# LOAD MODEL
# -----------------------------
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map=device
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Keep original for alerts/visualization
df_original = df.copy()

# Create proper continuous timestamps
df["timestamp"] = pd.date_range(
    start="2024-01-01",
    periods=len(df),
    freq="5min"
)

# Add required ID
df["id"] = 0

# Rename for Chronos
df = df.rename(columns={"crowd_count": "target"})

# -----------------------------
# NORMALIZATION (IMPORTANT)
# -----------------------------
mean = df["target"].mean()
std = df["target"].std()

df["target"] = (df["target"] - mean) / std

# -----------------------------
# CONTEXT + FUTURE
# -----------------------------
context_df = df.copy()

last_time = context_df["timestamp"].iloc[-1]

future_timestamps = pd.date_range(
    start=last_time,
    periods=PREDICTION_LENGTH + 1,
    freq="5min"
)[1:]

future_df = pd.DataFrame({
    "timestamp": future_timestamps,
    "id": 0,
    "time_of_day": [t.hour for t in future_timestamps],
    "hotspot_count": [df_original["hotspot_count"].iloc[-1]] * PREDICTION_LENGTH
})

# -----------------------------
# PREDICTION
# -----------------------------
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=PREDICTION_LENGTH,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target"
)

print("\nPrediction DataFrame:")
print(pred_df)

# -----------------------------
# DE-NORMALIZE OUTPUT
# -----------------------------
predicted_values = pred_df["0.5"].values * std + mean
upper_values = pred_df["0.9"].values * std + mean
lower_values = pred_df["0.1"].values * std + mean

current_count = df_original["crowd_count"].iloc[-1]

# -----------------------------
# ALERT SYSTEM (IMPROVED)
# -----------------------------
print("\nALERT ANALYSIS")

if any(predicted_values > MAX_CAPACITY):
    print("Overcrowding predicted!")

elif any(upper_values > MAX_CAPACITY):
    print("Possible overcrowding (uncertain but risky)")

elif max(predicted_values) > 0.8 * MAX_CAPACITY:
    print("Approaching capacity")

elif (predicted_values[0] - current_count) > SURGE_THRESHOLD:
    print("Rapid crowd surge detected")

else:
    print("Crowd levels are stable.")

# -----------------------------
# RISK SCORE (ADVANCED)
# -----------------------------
risk_score = max(upper_values) / MAX_CAPACITY

if risk_score > 1:
    print("HIGH RISK")
elif risk_score > 0.8:
    print("MEDIUM RISK")
else:
    print("LOW RISK")

# -----------------------------
# VISUALIZATION
# -----------------------------
history = df_original["crowd_count"].values[-50:]

history_x = list(range(len(history)))
future_x = list(range(len(history), len(history) + len(predicted_values)))

plt.figure(figsize=(10, 5))

plt.plot(history_x, history, label="Past Crowd", linewidth=2)
plt.plot(future_x, predicted_values, linestyle="dashed", label="Predicted Crowd")

# confidence band
plt.fill_between(future_x, lower_values, upper_values, alpha=0.2, label="Uncertainty")

plt.axhline(y=MAX_CAPACITY, linestyle="--", label="Max Capacity")

plt.title("Crowd Forecast (Chronos-2)")
plt.xlabel("Time Steps")
plt.ylabel("Crowd Count")
plt.legend()

plt.show()