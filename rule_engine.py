def kettle_rules(slope):
    """
    Evaluate kettle health based on degradation slope.
    
    Slope = increase in boil duration per cycle (seconds/cycle)
    
    Demo-tuned thresholds:
    - slope < 0.5: Healthy - no significant scaling
    - slope 0.5-1.5: Mild Scaling - descale soon  
    - slope > 1.5: Severe Scaling - descale immediately
    
    Expected demo behavior (single click):
    - 1× (10 cycles): slope ~2.0 → Severe (or start fresh for Mild)
    - 2× (20 cycles): slope ~2.0 → Severe
    - 3× (30 cycles): slope ~2.0 → Severe
    - After descale: slope ~0 → Healthy
    """
    if slope < 0.5:
        return {
            "status": "Healthy",
            "alert": None
        }
    elif slope < 1.5:
        return {
            "status": "Mild Scaling",
            "alert": "Mild scaling detected — Descale soon"
        }
    else:
        return {
            "status": "Severe Scaling",
            "alert": "Severe scaling detected — Efficiency reduced. Descale immediately"
        }


def chimney_rules(slope, recent_clean_events, auto_clean_enabled=True):
    """
    Evaluate chimney health based on motor current slope.
    
    Slope = increase in motor current per cycle (Amps/cycle)
    
    Demo-tuned thresholds:
    - slope < 0.005: Healthy
    - slope 0.005-0.02: Grease Buildup - needs cleaning
    - slope > 0.02: Severe - auto-clean may not be enough
    
    Returns different alerts based on auto_clean_enabled flag.
    """
    if slope < 0.005:
        return {
            "status": "Healthy",
            "alert": None,
            "auto_clean": False,
            "needs_cleaning": False
        }
    elif slope < 0.02:
        # Grease buildup detected
        if auto_clean_enabled:
            return {
                "status": "Grease Buildup",
                "alert": "Grease detected — AI will trigger auto-clean",
                "auto_clean": True,
                "needs_cleaning": True
            }
        else:
            return {
                "status": "Grease Buildup",
                "alert": "Grease detected — Click 'Manual Auto-Clean' to clean",
                "auto_clean": False,
                "needs_cleaning": True
            }
    else:
        # Severe buildup
        return {
            "status": "Severe Clogging",
            "alert": "Heavy grease buildup — Auto-clean may not be sufficient. Schedule service.",
            "auto_clean": False,
            "needs_cleaning": True
        }
