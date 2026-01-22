from flask import Flask, request, jsonify, render_template
from data_logger import log_to_csv
import pandas as pd
import numpy as np
import random
import time
from rule_engine import kettle_rules, chimney_rules
from simulator.kettle_sim import KettleSimulator
from simulator.chimney_sim import ChimneySimulator
from ml_model import ml_model  # New ML model


# App & Global State
app = Flask(__name__)

kettle_sim = KettleSimulator()
chimney_sim = ChimneySimulator()

kettle_on = True  # Default to ON so simulation works
chimney_on = True  # Default to ON so simulation works
auto_clean_enabled = True


# Utility Functions
def process_packet(data):
    device_model = data["device_model"]
    payload = data["payload"]
    timestamp = data.get("timestamp")

    if device_model == "Smart-1.7":  # Kettle
        row = {
            "device_model": device_model,
            "timestamp": timestamp,
            "effort_metric": payload["boil_duration"],
            "efficiency_metric": payload["energy_used"],
            "usage_metric": payload["cycle_count"],
            "maintenance_event": 0
        }

    elif device_model == "Oscar-600":  # Chimney
        row = {
            "device_model": device_model,
            "timestamp": timestamp,
            "effort_metric": payload["avg_current"],
            "efficiency_metric": payload["peak_current"],
            "usage_metric": payload["runtime"],
            "maintenance_event": payload["auto_clean_flag"]
        }

    else:
        raise ValueError("Unknown device model")

    log_to_csv(row)


def load_telemetry():
    try:
        df = pd.read_csv("data/telemetry_log.csv")
        df = df[pd.to_numeric(df["timestamp"], errors="coerce").notnull()]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="s")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def compute_slope(df):
    """
    Compute degradation slope using only RECENT data points.
    This ensures descaling resets the slope when new healthy data comes in.
    """
    # Only use last 20 data points for slope calculation
    # This makes the slope responsive to recent changes (like descaling)
    RECENT_WINDOW = 20
    recent_df = df.tail(RECENT_WINDOW).copy()
    
    if len(recent_df) < 3:
        return 0.0

    recent_df["t_index"] = range(len(recent_df))
    slope, _ = np.polyfit(recent_df["t_index"], recent_df["effort_metric"], 1)
    return slope


# API: Device Health Status (Enhanced with ML Model)
@app.route("/api/device_status/<device_model>", methods=["GET"])
def device_status(device_model):
    df = load_telemetry()
    
    if df.empty:
        return jsonify({"error": "No data found"}), 404
    
    device_df = df[df["device_model"] == device_model]

    if device_df.empty:
        return jsonify({"error": "No data found"}), 404

    # Use new ML model for comprehensive analysis
    analysis = ml_model.analyze(df, device_model)
    
    # Legacy slope calculation for backward compatibility
    device_df = device_df.copy()
    slope = compute_slope(device_df)
    recent_events = int(device_df["maintenance_event"].tail(10).sum())

    # Apply AI rules (using linear slope for status classification)
    if "Smart" in device_model:
        result = kettle_rules(slope)
    else:
        # Pass auto_clean_enabled to get appropriate alerts
        result = chimney_rules(slope, recent_events, auto_clean_enabled)

    # Override status based on health score if device is degraded but slope is stable
    health_score = analysis["health_score"]
    status = result.get("status", "Unknown")
    alert = result.get("alert", "")
    
    # IMPORTANT: Auto-clean decision is based ONLY on health score, not slope
    # This prevents auto-clean from triggering too early
    auto_clean = False  # Start with False, only enable based on health score
    
    # Auto-clean only triggers at severe degradation (< 30% health)
    if health_score < 30:
        if "Smart" in device_model:
            status = "Severe Scaling"
            alert = "Severe degradation — Descale immediately!"
        else:
            status = "Severe Grease Buildup"
            alert = "Severe grease buildup — AI Auto-clean activating!"
            auto_clean = True  # Only auto-clean when severely degraded (< 30%)
    elif health_score < 50:
        if "Smart" in device_model:
            status = "Mild Scaling"
            alert = "Moderate degradation — Descale soon"
        else:
            status = "Grease Buildup"
            alert = "Grease buildup detected — Consider manual cleaning"
            # NO auto_clean - just a warning
    # If health >= 50%, keep original status from rule engine (likely "Healthy")

    # Build response with enhanced ML data
    response = {
        "device_model": device_model,
        "health": status,
        "slope": round(slope, 4),
        "alert": alert,
        "trend": analysis["trend"],
        "auto_clean_enabled": auto_clean_enabled if "Oscar" in device_model else None,
        "trigger_auto_clean": auto_clean and auto_clean_enabled,
        
        # New ML model data
        "ml_analysis": {
            "acceleration": analysis["acceleration"],
            "health_score": analysis["health_score"],
            "anomaly": analysis["anomaly"],
            "rul": analysis["rul"]
        }
    }

    return jsonify(response)



# Toggle Auto-Clean
@app.route("/chimney/auto_clean/toggle", methods=["POST"])
def toggle_auto_clean():
    global auto_clean_enabled
    auto_clean_enabled = not auto_clean_enabled
    return jsonify({"auto_clean_enabled": auto_clean_enabled})



# Device ON / OFF
@app.route("/kettle/on", methods=["POST"])
def kettle_on_route():
    global kettle_on
    kettle_on = True
    return jsonify({"status": "kettle ON"})


@app.route("/kettle/off", methods=["POST"])
def kettle_off_route():
    global kettle_on
    kettle_on = False
    return jsonify({"status": "kettle OFF"})


@app.route("/chimney/on", methods=["POST"])
def chimney_on_route():
    global chimney_on
    chimney_on = True
    return jsonify({"status": "chimney ON"})


@app.route("/chimney/off", methods=["POST"])
def chimney_off_route():
    global chimney_on
    chimney_on = False
    return jsonify({"status": "chimney OFF"})


# Simulation Endpoints
@app.route("/simulate/kettle", methods=["POST"])
def simulate_kettle():
    if not kettle_on:
        return jsonify({"error": "Kettle is OFF"}), 400

    cycles = int(request.args.get("cycles", 1))
    for _ in range(cycles):
        packet = kettle_sim.boil_cycle()
        process_packet(packet)

    return jsonify({"status": f"{cycles} kettle cycles simulated"})


@app.route("/simulate/kettle/descale", methods=["POST"])
def simulate_kettle_descale():
    kettle_sim.descale()
    
    # Log 15 healthy data points after descaling
    # Since slope uses last 20 points, this ensures the slope drops to near 0
    # All readings at base healthy time (~175s) = flat line = slope ~0
    for _ in range(15):
        healthy_packet = {
            "device_type": "kettle",
            "device_model": "Smart-1.7",
            "timestamp": time.time(),
            "payload": {
                "boil_duration": 175 + int(random.uniform(-1, 1)),  # base healthy time with minimal noise
                "energy_used": 0.14,
                "cycle_count": 0,
                "temp_summary": "normal_rise"
            }
        }
        process_packet(healthy_packet)
    
    return jsonify({
        "status": "descaled",
        "message": "Kettle cleaned and reset. Healthy readings logged."
    })


@app.route("/simulate/chimney", methods=["POST"])
def simulate_chimney():
    if not chimney_on:
        return jsonify({"error": "Chimney is OFF"}), 400

    cycles = int(request.args.get("cycles", 1))
    for _ in range(cycles):
        packet = chimney_sim.run_cycle()
        process_packet(packet)

    return jsonify({"status": f"{cycles} chimney cycles simulated"})


@app.route("/simulate/chimney/auto_clean", methods=["POST"])
def simulate_chimney_auto_clean():
    """Manual auto-clean button - resets grease and logs healthy readings."""
    packet = chimney_sim.auto_clean()
    process_packet(packet)
    
    # Log 15 healthy readings to reset the slope
    for _ in range(15):
        healthy_packet = {
            "device_type": "chimney",
            "device_model": "Oscar-600",
            "timestamp": time.time(),
            "payload": {
                "avg_current": 1.0 + random.uniform(-0.01, 0.01),
                "peak_current": 1.15,
                "speed_level": 1,
                "runtime": 0,
                "auto_clean_flag": 1
            }
        }
        process_packet(healthy_packet)
    
    return jsonify({
        "status": "auto_cleaned",
        "message": "Filter cleaned! Healthy readings logged."
    })


# UI & Debug
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/debug/routes")
def debug_routes():
    return jsonify([str(rule) for rule in app.url_map.iter_rules()])



# Main
if __name__ == "__main__":
    app.run(port=8000, debug=True)
