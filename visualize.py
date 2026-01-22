import pandas as pd
import matplotlib.pyplot as plt
import os

# Load telemetry data
CSV_PATH = "data/telemetry_log.csv"

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found")
    exit(1)

df = pd.read_csv(CSV_PATH)

# Filter out invalid timestamps
df = df[pd.to_numeric(df["timestamp"], errors="coerce").notnull()]

if df.empty:
    print("Error: No valid data in telemetry log")
    exit(1)

# Convert timestamp to readable time
df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="s")

# -------------------------------
# KETTLE VISUALIZATION
# -------------------------------
kettle_df = df[df["device_model"] == "Smart-1.7"]

if not kettle_df.empty:
    plt.figure()
    plt.plot(kettle_df["timestamp"], kettle_df["effort_metric"])
    plt.title("Kettle Boil Duration Over Time")
    plt.xlabel("Time")
    plt.ylabel("Boil Duration (seconds)")
    plt.show()
else:
    print("No kettle data available for visualization")

# -------------------------------
# CHIMNEY VISUALIZATION
# -------------------------------
chimney_df = df[df["device_model"] == "Oscar-600"]

if not chimney_df.empty:
    plt.figure()
    plt.plot(chimney_df["timestamp"], chimney_df["effort_metric"])
    plt.title("Chimney Motor Current Over Time")
    plt.xlabel("Time")
    plt.ylabel("Avg Motor Current (Amps)")
    plt.show()
else:
    print("No chimney data available for visualization")
