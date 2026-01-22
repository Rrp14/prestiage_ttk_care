# ai_service/simulator/kettle_sim.py

import random
import time

class KettleSimulator:
    """
    Simulates physical behavior of a smart electric kettle.
    Tracks scaling buildup and its effect on performance.
    
    Demo-tuned for quick showcase:
    - 1 click at 1× (10 boils): Mild scaling starts
    - 1 click at 2× (20 boils): Clear mild scaling  
    - 1 click at 3× (30 boils): Severe scaling
    """

    def __init__(self):
        # ---- Physical state ----
        self.scale_level = 0.0        # represents mineral buildup
        self.base_boil_time = 175     # seconds (healthy kettle)
        self.base_energy = 0.14       # kWh (healthy kettle)
        self.cycle_count = 0          # track total cycles

    def boil_cycle(self):
        """
        Simulate one boil cycle.
        Each cycle increases scale buildup significantly for demo visibility.
        """
        self.cycle_count += 1
        
        # Scale builds up - each cycle adds 2 seconds to boil time
        # This creates a clear upward slope visible in the graph
        self.scale_level += 2.0

        # Boil time increases with scale
        # At 30 cycles: 175 + 60 = 235 seconds (very noticeable)
        boil_duration = int(
            self.base_boil_time
            + self.scale_level
            + random.uniform(-1, 1)   # minimal noise for clear trend
        )

        # Energy usage increases with scale
        energy_used = round(
            self.base_energy + self.scale_level * 0.003,
            3
        )

        # Temperature curve classification
        if self.scale_level > 40:
            temp_summary = "very_slow_rise"
        elif self.scale_level > 20:
            temp_summary = "slow_rise"
        else:
            temp_summary = "normal_rise"

        return {
            "device_type": "kettle",
            "device_model": "Smart-1.7",
            "timestamp": time.time(),
            "payload": {
                "boil_duration": boil_duration,
                "energy_used": energy_used,
                "cycle_count": self.cycle_count,
                "temp_summary": temp_summary
            }
        }

    def descale(self):
        """
        Simulate descaling action.
        Resets physical state to healthy.
        """
        self.scale_level = 0.0
        self.cycle_count = 0
