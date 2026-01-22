import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load and clean data
CSV_PATH = "data/telemetry_log.csv"

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found")
    exit(1)

df = pd.read_csv(CSV_PATH)

df = df[pd.to_numeric(df["timestamp"], errors="coerce").notnull()]

if df.empty:
    print("Error: No valid data in telemetry log")
    exit(1)

df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="s")

# -------------------------
# KETTLE REGRESSION
# -------------------------
kettle_df = df[df["device_model"] == "Smart-1.7"].copy()

if len(kettle_df) >= 2:
    # Use device-specific index for regression
    kettle_df["t_index"] = range(len(kettle_df))
    
    x_k = kettle_df["t_index"]
    y_k = kettle_df["effort_metric"]

    k_slope, k_intercept = np.polyfit(x_k, y_k, 1)
    k_trend = k_slope * x_k + k_intercept

    plt.figure()
    plt.plot(kettle_df["timestamp"], y_k, alpha=0.3, label="Raw")
    plt.plot(kettle_df["timestamp"], k_trend, linewidth=2, label="Regression Line")
    plt.title(f"Kettle Degradation Trend (slope={k_slope:.3f})")
    plt.xlabel("Time")
    plt.ylabel("Boil Duration (seconds)")
    plt.legend()
    plt.show()
    
    print("Kettle slope:", k_slope)
else:
    print("Not enough kettle data for regression")

# -------------------------
# CHIMNEY REGRESSION
# -------------------------
chimney_df = df[df["device_model"] == "Oscar-600"].copy()

if len(chimney_df) >= 2:
    # Use device-specific index for regression
    chimney_df["t_index"] = range(len(chimney_df))
    
    x_c = chimney_df["t_index"]
    y_c = chimney_df["effort_metric"]

    c_slope, c_intercept = np.polyfit(x_c, y_c, 1)
    c_trend = c_slope * x_c + c_intercept

    plt.figure()
    plt.plot(chimney_df["timestamp"], y_c, alpha=0.3, label="Raw")
    plt.plot(chimney_df["timestamp"], c_trend, linewidth=2, label="Regression Line")
    plt.title(f"Chimney Degradation Trend (slope={c_slope:.4f})")
    plt.xlabel("Time")
    plt.ylabel("Avg Motor Current (Amps)")
    plt.legend()
    plt.show()
    
    print("Chimney slope:", c_slope)
else:
    print("Not enough chimney data for regression")
