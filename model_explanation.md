# TTK Care AI - Complete Technical Documentation

## Overview

This document explains the complete data flow and ML pipeline for the TTK Care Predictive Maintenance System. The system monitors IoT appliances (Smart Kettle and Kitchen Chimney) and uses machine learning to predict maintenance needs.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Simulation Layer](#2-data-simulation-layer)
3. [Data Logging Pipeline](#3-data-logging-pipeline)
4. [ML Model Analysis](#4-ml-model-analysis)
5. [Rule Engine](#5-rule-engine)
6. [API Response Construction](#6-api-response-construction)
7. [Frontend Visualization](#7-frontend-visualization)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TTK Care AI System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Simulators  │───▶│ Data Logger  │───▶│   ML Model   │───▶│    API    │ │
│  │ kettle_sim   │    │  CSV Storage │    │  Polynomial  │    │  Flask    │ │
│  │ chimney_sim  │    │              │    │  Regression  │    │  REST     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Physical     │    │ telemetry_   │    │ Health Score │    │ Dashboard │ │
│  │ Simulation   │    │ log.csv      │    │ Slope, RUL   │    │ Chart.js  │ │
│  │ (Degradation)│    │              │    │ Anomaly      │    │ HTML/JS   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Simulation Layer

### 2.1 Smart Kettle Simulator (`simulator/kettle_sim.py`)

The kettle simulator models **mineral scale buildup** that affects boiling performance.

#### Physical Model:

```python
class KettleSimulator:
    def __init__(self):
        self.scale_level = 0.0        # Mineral buildup (0-100)
        self.base_boil_time = 175     # Healthy boil time (seconds)
        self.base_energy = 0.14       # Healthy energy usage (kWh)
```

#### Degradation Formula:

```python
def boil_cycle(self):
    # Each cycle adds 2 seconds worth of scale
    self.scale_level += 2.0
    
    # Boil time increases linearly with scale
    boil_duration = self.base_boil_time + self.scale_level + noise
    
    # Energy usage also increases
    energy_used = self.base_energy + (self.scale_level * 0.003)
```

#### Output Packet:

```python
{
    "device_type": "kettle",
    "device_model": "Smart-1.7",
    "timestamp": 1706000000.0,
    "payload": {
        "boil_duration": 195,      # seconds (effort_metric)
        "energy_used": 0.20,       # kWh (efficiency_metric)
        "cycle_count": 10,         # usage_metric
        "temp_summary": "slow_rise"
    }
}
```

#### Degradation Curve:

```
Boil Time (s)
     │
 235 │                                    ████████  ← Severe (need descale)
     │                              ██████
 210 │                        ██████
     │                  ██████
 195 │            ██████                             ← Mild scaling
     │      ██████
 175 │██████                                         ← Healthy baseline
     └────────────────────────────────────────────▶
         0    10    20    30    40    50   Cycles
```

---

### 2.2 Kitchen Chimney Simulator (`simulator/chimney_sim.py`)

The chimney simulator models **grease buildup** that increases motor load.

#### Physical Model:

```python
class ChimneySimulator:
    def __init__(self):
        self.grease_level = 0.0       # Grease accumulation (0-20)
        self.base_current = 1.0       # Healthy motor current (Amps)
        self.max_grease = 20.0        # Saturation point
```

#### Degradation Formula:

```python
def run_cycle(self, speed_level=2):
    # Grease builds with cooking intensity
    grease_increment = 0.3 * speed_level
    self.grease_level = min(self.grease_level + grease_increment, self.max_grease)
    
    # Motor current increases with grease (resistance)
    avg_current = self.base_current + (self.grease_level * 0.03) + noise
```

#### Output Packet:

```python
{
    "device_type": "chimney",
    "device_model": "Oscar-600",
    "timestamp": 1706000000.0,
    "payload": {
        "avg_current": 1.25,       # Amps (effort_metric)
        "peak_current": 1.40,      # Amps (efficiency_metric)
        "runtime": 30,             # cycles (usage_metric)
        "auto_clean_flag": 0       # maintenance_event
    }
}
```

---

## 3. Data Logging Pipeline

### 3.1 Packet Processing (`app.py` → `process_packet()`)

Converts device-specific payloads to a **unified schema**:

```python
def process_packet(data):
    device_model = data["device_model"]
    payload = data["payload"]
    
    if device_model == "Smart-1.7":  # Kettle
        row = {
            "device_model": device_model,
            "timestamp": data["timestamp"],
            "effort_metric": payload["boil_duration"],      # PRIMARY METRIC
            "efficiency_metric": payload["energy_used"],
            "usage_metric": payload["cycle_count"],
            "maintenance_event": 0
        }
    
    elif device_model == "Oscar-600":  # Chimney
        row = {
            "device_model": device_model,
            "timestamp": data["timestamp"],
            "effort_metric": payload["avg_current"],        # PRIMARY METRIC
            "efficiency_metric": payload["peak_current"],
            "usage_metric": payload["runtime"],
            "maintenance_event": payload["auto_clean_flag"]
        }
    
    log_to_csv(row)
```

### 3.2 CSV Storage (`data_logger.py`)

Data is appended to `data/telemetry_log.csv`:

```csv
device_model,timestamp,effort_metric,efficiency_metric,usage_metric,maintenance_event
Smart-1.7,1706000001.0,177,0.146,1,0
Smart-1.7,1706000002.0,179,0.152,2,0
Smart-1.7,1706000003.0,181,0.158,3,0
Oscar-600,1706000004.0,1.02,1.15,1,0
Oscar-600,1706000005.0,1.05,1.18,2,0
```

### 3.3 Key Metric: `effort_metric`

| Device | effort_metric | Unit | Meaning |
|--------|--------------|------|---------|
| Kettle | boil_duration | seconds | Time to boil water |
| Chimney | avg_current | Amps | Motor current draw |

**Why this metric?**
- Directly measurable by IoT sensors
- Monotonically increases with degradation
- Clear physical interpretation

---

## 4. ML Model Analysis

### 4.1 Model Architecture (`ml_model.py`)

The `PredictiveMaintenanceModel` class combines multiple techniques:

```python
class PredictiveMaintenanceModel:
    def __init__(self, window_size=20, poly_degree=2):
        self.window_size = 20      # Sliding window for recent data
        self.poly_degree = 2       # Polynomial degree (quadratic)
        self.anomaly_threshold = 2.5  # Z-score threshold
        
        # Device-specific thresholds
        self.failure_thresholds = {
            "Smart-1.7": 220,   # Kettle: 220s = severe scaling
            "Oscar-600": 1.5    # Chimney: 1.5A = severe grease
        }
        
        self.healthy_baselines = {
            "Smart-1.7": 175,   # Kettle: 175s = healthy
            "Oscar-600": 1.0    # Chimney: 1.0A = healthy
        }
```

---

### 4.2 Polynomial Regression (Slope & Acceleration)

#### Technique: Least Squares Polynomial Fitting

```python
def compute_polynomial_slope(self, df):
    recent_df = df.tail(self.window_size)  # Last 20 data points
    
    x = np.arange(len(recent_df))          # [0, 1, 2, ..., 19]
    y = recent_df["effort_metric"].values  # [177, 179, 181, ...]
    
    # Fit quadratic polynomial: y = ax² + bx + c
    coeffs = np.polyfit(x, y, 2)
    # coeffs = [a, b, c]
    
    acceleration = coeffs[0]   # 'a' - quadratic term
    linear_slope = coeffs[1]   # 'b' - linear term
    
    return linear_slope, acceleration
```

#### Mathematical Formula:

$$y = ax^2 + bx + c$$

Where:
- $a$ = **Acceleration** (how fast degradation speeds up)
- $b$ = **Slope** (rate of degradation per cycle)
- $c$ = **Intercept** (baseline value)

#### Interpretation:

| Coefficient | Value | Meaning |
|-------------|-------|---------|
| Slope (b) | +2.0 | Degrading 2 units per cycle |
| Slope (b) | 0 | Stable (no change) |
| Slope (b) | -0.5 | Improving (after maintenance) |
| Accel (a) | +0.1 | Degradation speeding up |
| Accel (a) | 0 | Linear degradation |
| Accel (a) | -0.05 | Degradation slowing (plateau) |

#### Why Polynomial Degree 2?

- **Degree 1 (Linear)**: Assumes constant degradation rate
- **Degree 2 (Quadratic)**: Captures accelerating/decelerating degradation
- **Degree 3+**: Overfitting risk with small windows

---

### 4.3 Health Score Calculation

#### Formula:

```python
def compute_health_score(self, df, device_model):
    current_value = df["effort_metric"].iloc[-1]  # Latest reading
    baseline = self.healthy_baselines[device_model]
    threshold = self.failure_thresholds[device_model]
    
    # Linear interpolation between baseline (100%) and threshold (0%)
    degradation_range = threshold - baseline
    current_degradation = current_value - baseline
    
    health_score = 100 - (current_degradation / degradation_range * 100)
    health_score = max(0, min(100, health_score))  # Clamp to 0-100
    
    return int(health_score)
```

#### Mathematical Formula:

$$\text{Health Score} = 100 \times \left(1 - \frac{\text{current} - \text{baseline}}{\text{threshold} - \text{baseline}}\right)$$

#### Example Calculation (Kettle):

```
baseline = 175s, threshold = 220s, current = 195s

health_score = 100 × (1 - (195 - 175) / (220 - 175))
             = 100 × (1 - 20 / 45)
             = 100 × (1 - 0.44)
             = 100 × 0.56
             = 56%
```

#### Visual Representation:

```
Health Score
100% ├────────────────┤ Healthy (175s)
     │████████████████│
 75% │████████████    │
     │████████████    │
 50% │████████        │ ← 195s (56%)
     │████████        │
 25% │████            │
     │████            │
  0% ├────────────────┤ Failure (220s)
```

---

### 4.4 Anomaly Detection (Z-Score Method)

#### Technique: Statistical Outlier Detection

```python
def detect_anomaly(self, df):
    recent_df = df.tail(self.window_size)
    
    # Rolling statistics (window=10)
    rolling_mean = recent_df["effort_metric"].rolling(10).mean()
    rolling_std = recent_df["effort_metric"].rolling(10).std()
    
    # Latest value
    latest_value = recent_df["effort_metric"].iloc[-1]
    
    # Z-score calculation
    z_score = (latest_value - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
    
    # Threshold check
    is_anomaly = abs(z_score) > 2.5
    
    return {"is_anomaly": is_anomaly, "z_score": z_score}
```

#### Mathematical Formula:

$$Z = \frac{x - \mu}{\sigma}$$

Where:
- $x$ = current value
- $\mu$ = rolling mean (last 10 points)
- $\sigma$ = rolling standard deviation

#### Threshold: |Z| > 2.5

| Z-Score Range | Probability | Interpretation |
|---------------|-------------|----------------|
| -1 to +1 | 68% | Normal variation |
| -2 to +2 | 95% | Expected range |
| -2.5 to +2.5 | 98.8% | Within bounds |
| **> 2.5 or < -2.5** | **1.2%** | **ANOMALY** |

#### Use Cases:

- **Sudden spike**: Sensor malfunction or rapid failure
- **Sudden drop**: Sensor reset or unexpected repair
- **Both require investigation**

---

### 4.5 Remaining Useful Life (RUL) Prediction

#### Technique: Polynomial Extrapolation

```python
def predict_remaining_life(self, df, device_model):
    recent_df = df.tail(self.window_size)
    x = np.arange(len(recent_df))
    y = recent_df["effort_metric"].values
    
    # Fit polynomial
    coeffs = np.polyfit(x, y, 2)
    
    failure_threshold = self.failure_thresholds[device_model]
    
    # Extrapolate into future
    for future_x in range(1, 1000):
        future_y = np.polyval(coeffs, len(recent_df) + future_x)
        if future_y >= failure_threshold:
            return {"cycles_to_failure": future_x}
    
    return {"cycles_to_failure": None}  # No failure in foreseeable future
```

#### Visualization:

```
effort_metric
      │
  220 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─●  ← Failure threshold
      │                           ●●●●●
      │                     ●●●●●
  195 │               ●●●●●
      │         ●●●●●
      │   ●●●●●
  175 │●●●
      └────────────────────────────────────▶
          Now              RUL=25 cycles   Future
```

---

### 4.6 Complete Analysis Pipeline

```python
def analyze(self, df, device_model):
    device_df = df[df["device_model"] == device_model]
    
    # 1. Polynomial Regression
    slope, acceleration, trend_line = self.compute_polynomial_slope(device_df)
    
    # 2. Anomaly Detection
    anomaly = self.detect_anomaly(device_df)
    
    # 3. RUL Prediction
    rul = self.predict_remaining_life(device_df, device_model)
    
    # 4. Health Score
    health_score = self.compute_health_score(device_df, device_model)
    
    # 5. Trend for chart (moving average)
    device_df["ma"] = device_df["effort_metric"].rolling(10).mean()
    trend = device_df["ma"].tail(15).tolist()
    
    return {
        "slope": slope,
        "acceleration": acceleration,
        "anomaly": anomaly,
        "rul": rul,
        "health_score": health_score,
        "trend": trend
    }
```

---

## 5. Rule Engine

### 5.1 Kettle Rules (`rule_engine.py`)

```python
def kettle_rules(slope):
    if slope < 0.5:
        return {"status": "Healthy", "alert": None}
    elif slope < 1.5:
        return {"status": "Mild Scaling", "alert": "Descale soon"}
    else:
        return {"status": "Severe Scaling", "alert": "Descale immediately"}
```

### 5.2 Chimney Rules

```python
def chimney_rules(slope, recent_events, auto_clean_enabled):
    if slope < 0.005:
        return {"status": "Healthy", "auto_clean": False}
    elif slope < 0.02:
        return {"status": "Grease Buildup", "auto_clean": auto_clean_enabled}
    else:
        return {"status": "Severe Buildup", "auto_clean": True}
```

### 5.3 Health Score Override (`app.py`)

The rule engine uses **slope**, but we also check **health score**:

```python
# Override based on health score (catches stable but degraded devices)
if health_score < 30:
    status = "Severe"
    auto_clean = True   # Only trigger auto-clean when severely degraded
elif health_score < 50:
    status = "Moderate"
    # Warning only, no auto-clean
```

---

## 6. API Response Construction

### 6.1 Device Status Endpoint

```python
@app.route("/api/device_status/<device_model>")
def device_status(device_model):
    df = load_telemetry()
    
    # ML Analysis
    analysis = ml_model.analyze(df, device_model)
    
    # Rule Engine
    slope = compute_slope(device_df)
    result = kettle_rules(slope) if "Smart" in device_model else chimney_rules(...)
    
    # Build Response
    response = {
        "device_model": device_model,
        "health": result["status"],
        "slope": slope,
        "alert": result["alert"],
        "trend": analysis["trend"],
        "ml_analysis": {
            "acceleration": analysis["acceleration"],
            "health_score": analysis["health_score"],
            "anomaly": analysis["anomaly"],
            "rul": analysis["rul"]
        }
    }
    
    return jsonify(response)
```

### 6.2 Sample API Response

```json
{
    "device_model": "Smart-1.7",
    "health": "Mild Scaling",
    "slope": 1.2345,
    "alert": "Moderate degradation — Descale soon",
    "trend": [180.5, 182.3, 184.1, 186.0, 187.8, ...],
    "auto_clean_enabled": null,
    "trigger_auto_clean": false,
    "ml_analysis": {
        "acceleration": -0.0234,
        "health_score": 56,
        "anomaly": {
            "is_anomaly": false,
            "z_score": 0.82,
            "message": null
        },
        "rul": {
            "cycles_to_failure": 25,
            "confidence": "medium",
            "message": "~25 cycles until maintenance needed"
        }
    }
}
```

---

## 7. Frontend Visualization

### 7.1 Health Bar

```javascript
function updateHealthBar(score, barElement, scoreElement) {
    barElement.style.width = score + '%';
    scoreElement.innerText = score + '%';
    
    // Color gradient based on health
    if (score >= 70) {
        barElement.style.background = 'green';
    } else if (score >= 40) {
        barElement.style.background = 'yellow';
    } else {
        barElement.style.background = 'red';
    }
}
```

### 7.2 Trend Chart (Chart.js)

```javascript
function drawChart(canvasId, data) {
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: [...Array(data.length).keys()],
            datasets: [{
                label: 'Effort Metric (Moving Avg)',
                data: data,
                borderColor: '#3b82f6',
                tension: 0.4
            }]
        }
    });
}
```

---

## Summary: Complete Data Flow

```
1. USER ACTION
   └── Click "Simulate Boil" (30 cycles)

2. SIMULATION
   └── kettle_sim.boil_cycle() × 30
       └── scale_level += 2.0 per cycle
       └── boil_duration = 175 + scale + noise

3. DATA LOGGING
   └── process_packet() extracts effort_metric
   └── log_to_csv() appends to telemetry_log.csv

4. ML ANALYSIS
   └── ml_model.analyze("Smart-1.7")
       ├── compute_polynomial_slope() → slope=2.0, accel=-0.01
       ├── detect_anomaly() → is_anomaly=false, z_score=0.5
       ├── predict_remaining_life() → rul=15 cycles
       └── compute_health_score() → health=33%

5. RULE ENGINE
   └── kettle_rules(slope=2.0) → "Severe Scaling"
   └── health_score override → confirms "Severe"

6. API RESPONSE
   └── JSON with all metrics

7. DASHBOARD UPDATE
   └── Health bar: 33% (red)
   └── Status: "Severe Scaling"
   └── Chart: upward trend
   └── RUL: "~15 cycles to failure"
```

---

## Techniques Summary

| Component | Technique | Why |
|-----------|-----------|-----|
| Degradation Model | Linear accumulation | Simple, physically meaningful |
| Trend Detection | Polynomial Regression (degree 2) | Captures acceleration |
| Anomaly Detection | Z-Score | Simple, interpretable |
| RUL Prediction | Polynomial Extrapolation | Projects future degradation |
| Health Score | Linear Interpolation | Easy to understand |
| Data Storage | CSV | Simple, human-readable |
| Visualization | Chart.js | Lightweight, responsive |

---

## Future Improvements

1. **LSTM/RNN** for time-series prediction
2. **Isolation Forest** for multivariate anomaly detection
3. **Kalman Filter** for noise reduction
4. **Ensemble Methods** for robust predictions
5. **Database** (SQLite/PostgreSQL) for production
6. **Real-time WebSocket** updates instead of polling

---

*Document generated for TTK Care AI Predictive Maintenance System*
*Version 2.0 - January 2026*
