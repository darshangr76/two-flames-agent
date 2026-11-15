"""
ZENITHFORGE - AstraFlame Data Loader
Enforces the strict distinction between Measured, Calculated, and Predicted data.
"""
import pandas as pd

def load_flex_baseline(experiment_id="FLEX-035"):
    """
    Simulates loading the genuine NASA FLEX dataset.
    Every value is tagged to maintain scientific integrity.
    """
    # In a real app, this reads from data/raw/nasa_flex/PSI-69.csv
    data = {
        "experiment_id": experiment_id,
        "investigation": "PSI-69 (FLEX)",
        "measurements": {
            "fuel": {"value": "Heptane", "type": "Measured", "source": "NASA PSI-69 Table"},
            "o2_pct": {"value": 21.0, "type": "Measured", "source": "NASA PSI-69 Table"},
            "pressure_atm": {"value": 1.0, "type": "Measured", "source": "NASA PSI-69 Table"},
            "droplet_mm": {"value": 3.91, "type": "Measured", "source": "NASA PSI-69 Table"}
        },
        "calculated_physics": {
            "burn_rate_mm2_s": {"value": 0.64, "type": "Calculated", "source": "Regression on FLEX-035 data"}
        },
        "ai_predictions": {} # To be filled by the ML model
    }
    return data

def attach_prediction(data, predicted_outcome, confidence):
    """Attaches AI predictions without overwriting measured data."""
    data["ai_predictions"] = {
        "predicted_outcome": {"value": predicted_outcome, "type": "Predicted", "confidence": confidence},
        "digital_twin_status": {"value": "Data-driven exploratory model", "type": "Predicted"}
    }
    return data

if __name__ == '__main__':
    import json
    record = load_flex_baseline()
    record = attach_prediction(record, "Cool Flame", "92%")
    print(json.dumps(record, indent=2))
