# ai_service/ml_model.py
"""
Enhanced ML Model for Predictive Maintenance

Models:
1. Polynomial Regression - Captures non-linear degradation
2. Z-Score Anomaly Detection - Detects sudden failures
3. Remaining Useful Life (RUL) Prediction - Predicts cycles to failure
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class PredictiveMaintenanceModel:
    """
    Enhanced ML model combining:
    - Polynomial regression for degradation trend
    - Z-score for anomaly detection
    - RUL (Remaining Useful Life) prediction
    """
    
    def __init__(self, window_size: int = 20, poly_degree: int = 2):
        self.window_size = window_size
        self.poly_degree = poly_degree
        
        # Thresholds for anomaly detection
        self.anomaly_threshold = 2.5  # Z-score threshold
        
        # Failure thresholds (when device needs maintenance)
        self.failure_thresholds = {
            "Smart-1.7": 220,    # Kettle: boil time > 220s = severe
            "Oscar-600": 1.5     # Chimney: current > 1.5A = severe (raised from 1.4)
        }
        
        # Base healthy values
        self.healthy_baselines = {
            "Smart-1.7": 175,    # Kettle: healthy boil time
            "Oscar-600": 1.0     # Chimney: healthy current (1.0A baseline)
        }
    
    def compute_polynomial_slope(self, df: pd.DataFrame) -> Tuple[float, float, np.ndarray]:
        """
        Fit polynomial regression and return:
        - Linear slope component (for backward compatibility)
        - Acceleration (quadratic coefficient)
        - Fitted trend line
        """
        if len(df) < 5:
            return 0.0, 0.0, np.array([])
        
        recent_df = df.tail(self.window_size).copy()
        x = np.arange(len(recent_df))
        
        # Ensure numeric values
        y = pd.to_numeric(recent_df["effort_metric"], errors="coerce").fillna(0).values
        
        # Fit polynomial (degree 2: y = ax² + bx + c)
        coeffs = np.polyfit(x, y, self.poly_degree)
        
        # coeffs = [a, b, c] for ax² + bx + c
        acceleration = float(coeffs[0])  # Quadratic term (how fast degradation speeds up)
        linear_slope = float(coeffs[1])  # Linear term
        
        # Generate fitted trend line
        trend_line = np.polyval(coeffs, x)
        
        return linear_slope, acceleration, trend_line
    
    def detect_anomaly(self, df: pd.DataFrame) -> Dict:
        """
        Detect anomalies using Z-score method.
        Returns anomaly status and details.
        """
        if len(df) < 10:
            return {
                "is_anomaly": False,
                "z_score": 0.0,
                "message": None
            }
        
        recent_df = df.tail(self.window_size).copy()
        
        # Ensure numeric values
        effort_numeric = pd.to_numeric(recent_df["effort_metric"], errors="coerce")
        
        # Calculate rolling statistics
        rolling_mean = effort_numeric.rolling(window=10).mean()
        rolling_std = effort_numeric.rolling(window=10).std()
        
        # Get latest value
        latest_value = float(effort_numeric.iloc[-1])
        latest_mean = float(rolling_mean.iloc[-1])
        latest_std = float(rolling_std.iloc[-1])
        
        # Avoid division by zero or NaN
        if latest_std == 0 or np.isnan(latest_std) or np.isnan(latest_value) or np.isnan(latest_mean):
            return {
                "is_anomaly": False,
                "z_score": 0.0,
                "message": None
            }
        
        # Calculate Z-score
        z_score = (latest_value - latest_mean) / latest_std
        
        is_anomaly = bool(abs(z_score) > self.anomaly_threshold)  # Convert to native Python bool
        
        message = None
        if is_anomaly:
            if z_score > 0:
                message = "⚠️ ANOMALY: Sudden spike detected! Possible sensor fault or rapid degradation."
            else:
                message = "⚠️ ANOMALY: Sudden drop detected! Possible sensor fault."
        
        return {
            "is_anomaly": is_anomaly,
            "z_score": float(round(z_score, 2)),  # Ensure native Python float
            "message": message
        }
    
    def predict_remaining_life(self, df: pd.DataFrame, device_model: str) -> Dict:
        """
        Predict Remaining Useful Life (RUL) - cycles until failure threshold.
        Uses polynomial extrapolation.
        """
        if len(df) < 10:
            return {
                "cycles_to_failure": None,
                "confidence": "low",
                "message": "Insufficient data for prediction"
            }
        
        recent_df = df.tail(self.window_size).copy()
        x = np.arange(len(recent_df))
        
        # Ensure numeric values
        y = pd.to_numeric(recent_df["effort_metric"], errors="coerce").fillna(0).values
        
        # Fit polynomial
        coeffs = np.polyfit(x, y, self.poly_degree)
        
        # Get failure threshold for this device
        failure_threshold = self.failure_thresholds.get(device_model, float(y[-1]) * 1.3)
        current_value = float(y[-1])
        
        # If already above threshold
        if current_value >= failure_threshold:
            return {
                "cycles_to_failure": 0,
                "confidence": "high",
                "message": "⚠️ Maintenance needed NOW!"
            }
        
        # Predict future values until we hit threshold
        cycles_to_failure = None
        for future_x in range(1, 1000):  # Look up to 1000 cycles ahead
            future_y = np.polyval(coeffs, len(recent_df) + future_x)
            if future_y >= failure_threshold:
                cycles_to_failure = future_x
                break
        
        if cycles_to_failure is None:
            return {
                "cycles_to_failure": None,
                "confidence": "low",
                "message": "No failure predicted in near future"
            }
        
        # Determine confidence based on data quality
        if len(recent_df) >= 15 and cycles_to_failure < 100:
            confidence = "high"
        elif len(recent_df) >= 10:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate message
        if cycles_to_failure <= 10:
            message = f"🔴 CRITICAL: ~{cycles_to_failure} cycles to failure!"
        elif cycles_to_failure <= 30:
            message = f"🟡 WARNING: ~{cycles_to_failure} cycles to failure"
        else:
            message = f"🟢 ~{cycles_to_failure} cycles until maintenance needed"
        
        return {
            "cycles_to_failure": cycles_to_failure,
            "confidence": confidence,
            "message": message
        }
    
    def compute_health_score(self, df: pd.DataFrame, device_model: str) -> int:
        """
        Compute overall health score (0-100).
        100 = Perfect health, 0 = Failure
        """
        if len(df) < 5:
            return 100
        
        # Ensure effort_metric is numeric
        effort_values = pd.to_numeric(df["effort_metric"], errors="coerce")
        current_value = float(effort_values.iloc[-1])
        
        # Handle NaN
        if np.isnan(current_value):
            return 100
            
        baseline = self.healthy_baselines.get(device_model, current_value)
        threshold = self.failure_thresholds.get(device_model, baseline * 1.3)
        
        # Calculate how far we are from baseline to threshold
        degradation_range = threshold - baseline
        current_degradation = current_value - baseline
        
        if degradation_range <= 0:
            return 100
        
        # Health score: 100 at baseline, 0 at threshold
        health_score = 100 - (current_degradation / degradation_range * 100)
        health_score = max(0, min(100, health_score))  # Clamp to 0-100
        
        return int(health_score)
    
    def analyze(self, df: pd.DataFrame, device_model: str) -> Dict:
        """
        Complete analysis combining all models.
        Returns comprehensive health assessment.
        """
        # Filter for specific device
        device_df = df[df["device_model"] == device_model].copy()
        
        if device_df.empty:
            return {
                "status": "No Data",
                "slope": 0,
                "acceleration": 0,
                "anomaly": {"is_anomaly": False, "z_score": 0, "message": None},
                "rul": {"cycles_to_failure": None, "confidence": "low", "message": "No data"},
                "health_score": 100,
                "trend": []
            }
        
        # 1. Polynomial Regression
        slope, acceleration, trend_line = self.compute_polynomial_slope(device_df)
        
        # 2. Anomaly Detection
        anomaly = self.detect_anomaly(device_df)
        
        # 3. Remaining Useful Life
        rul = self.predict_remaining_life(device_df, device_model)
        
        # 4. Health Score
        health_score = self.compute_health_score(device_df, device_model)
        
        # 5. Moving average trend for chart
        WINDOW = 10
        device_df["ma_effort"] = device_df["effort_metric"].rolling(WINDOW).mean()
        trend = [float(x) if not np.isnan(x) else 0.0 for x in device_df["ma_effort"].tail(15).tolist()]
        
        return {
            "slope": float(round(slope, 4)),
            "acceleration": float(round(acceleration, 4)),
            "anomaly": anomaly,
            "rul": rul,
            "health_score": int(health_score),
            "trend": trend
        }


# Singleton instance for use across the app
ml_model = PredictiveMaintenanceModel()
