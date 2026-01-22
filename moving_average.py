import pandas as pd
import matplotlib.pyplot as plt
import os

# Read CSV and drop header-like rows
CSV_PATH = "data/telemetry_log.csv"

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found")
    exit(1)

df = pd.read_csv(CSV_PATH)

# Remove rows where timestamp is not numeric
df = df[pd.to_numeric(df["timestamp"], errors="coerce").notnull()]

if df.empty:
    print("Error: No valid data in telemetry log")
    exit(1)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="s")

WINDOW = 5

# -------------------------
# KETTLE
# -------------------------
kettle_df = df[df["device_model"] == "Smart-1.7"].copy()

if not kettle_df.empty:
    kettle_df["ma_effort"] = kettle_df["effort_metric"].rolling(WINDOW).mean()

    plt.figure()
    plt.plot(kettle_df["timestamp"], kettle_df["effort_metric"], alpha=0.3, label="Raw")
    plt.plot(kettle_df["timestamp"], kettle_df["ma_effort"], linewidth=2, label="Moving Avg")
    plt.title("Kettle Boil Duration (Moving Average)")
    plt.xlabel("Time")
    plt.ylabel("Boil Duration (seconds)")
    plt.legend()
    plt.show()
else:
    print("No kettle data available")

# -------------------------
# CHIMNEY
# -------------------------
chimney_df = df[df["device_model"] == "Oscar-600"].copy()

if not chimney_df.empty:
    chimney_df["ma_effort"] = chimney_df["effort_metric"].rolling(WINDOW).mean()

    plt.figure()
    plt.plot(chimney_df["timestamp"], chimney_df["effort_metric"], alpha=0.3, label="Raw")
    plt.plot(chimney_df["timestamp"], chimney_df["ma_effort"], linewidth=2, label="Moving Avg")
    plt.title("Chimney Motor Current (Moving Average)")
    plt.xlabel("Time")
    plt.ylabel("Avg Motor Current (Amps)")
    plt.legend()
    plt.show()
else:
    print("No chimney data available")
