# ai_service/simulator/chimney_sim.py

import random
import time

class ChimneySimulator:
    """
    Simulates physical behavior of an IoT kitchen chimney.
    Tracks grease buildup and its effect on motor current.
    
    Demo-tuned for quick showcase:
    - 1 click at 3× (30 cycles): Clear grease buildup visible
    - Auto-clean resets to healthy
    """

    def __init__(self):
        # ---- Physical state ----
        self.grease_level = 0.0           # invisible grease accumulation
        self.base_current = 1.0           # Amps (clean filter)
        self.max_grease = 10.0            # saturation point
        self.cycle_count = 0

    def run_cycle(self, speed_level=2):
        """
        Simulate one chimney usage cycle.
        Grease builds with usage, motor current increases.
        
        Tuned: each cycle adds 0.02 Amps to current draw
        At 30 cycles: current goes from 1.0 to ~1.6 Amps (visible slope)
        """
        self.cycle_count += 1
        
        # Grease builds up - tuned for visible degradation
        grease_increment = 0.3 * speed_level  # Much faster buildup
        self.grease_level = min(
            self.grease_level + grease_increment,
            self.max_grease
        )

        # Current draw increases with grease (0.02 Amps per cycle)
        avg_current = round(
            self.base_current
            + (self.grease_level * 0.03)
            + random.uniform(-0.01, 0.01),   # minimal noise for clear trend
            2
        )

        peak_current = round(
            avg_current + random.uniform(0.1, 0.2),
            2
        )

        runtime = self.cycle_count

        return {
            "device_type": "chimney",
            "device_model": "Oscar-600",
            "timestamp": time.time(),
            "payload": {
                "avg_current": avg_current,
                "peak_current": peak_current,
                "speed_level": speed_level,
                "runtime": runtime,
                "auto_clean_flag": 0
            }
        }

    def auto_clean(self):
        """
        Simulate thermal auto-clean.
        Resets grease level back to clean state.
        """
        self.grease_level = 0.0
        self.cycle_count = 0

        return {
            "device_type": "chimney",
            "device_model": "Oscar-600",
            "timestamp": time.time(),
            "payload": {
                "avg_current": self.base_current,
                "peak_current": round(self.base_current + 0.15, 2),
                "speed_level": 1,
                "runtime": 0,
                "auto_clean_flag": 1
            }
        }
