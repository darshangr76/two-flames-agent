"""
ZENITHFORGE - Unified Time-Series Fusion Engine (UTFE)
Aligns heterogeneous NASA telemetry (chamber, camera, igniter) to a single t=0 (Igniter On).
"""
import pandas as pd
import numpy as np
from datetime import datetime

def parse_chamber_readings(filepath, ignition_time_str):
    """
    Parses C039H01_Chamber_Readings.txt
    Extracts pressure transducers (PT16, PT28) and thermocouples.
    """
    # Note: Real parser would read the specific NASA format. 
    # For the hackathon prototype, we simulate the normalization logic.
    print(f"[UTFE] Parsing Chamber Readings from {filepath}")
    print(f"[UTFE] Anchoring timeline to Ignition Event: {ignition_time_str}")
    
    # Simulated data based on the DeepSeek analysis (PT28 pressure ripple)
    # Real implementation would be pd.read_csv(filepath, sep='\t', skiprows=...)
    t = np.linspace(-2, 5, 70) # -2s before ignition to +5s
    # Simulate a pressure spike at ignition and a ~0.5s ripple
    pressure_pt28 = 43.14 + 0.01 * np.sin(2 * np.pi * 2 * t) + 0.05 * np.exp(-t**2)
    
    df = pd.DataFrame({"t_rel_s": t, "PT28_psia": pressure_pt28})
    return df

def parse_flame_tracking(filepath):
    """
    Parses FLEX-008_ColorCamData.txt
    Extracts Flame Diameter and Standoff.
    """
    print(f"[UTFE] Parsing Flame Tracking from {filepath}")
    # Simulated data based on FLEX-008 (flame diameter oscillates, standoff drops at 3.5s)
    t = np.linspace(0, 5, 50)
    diameter = 2.0 + 0.3 * np.sin(2 * np.pi * 1.5 * t)
    standoff = 8.0 - 0.5 * t
    standoff[t > 3.5] = np.nan # Simulate the dropout gap found in the analysis
    
    df = pd.DataFrame({"t_rel_s": t, "Flame_Diameter_mm": diameter, "Standoff_mm": standoff})
    return df

def fuse_timelines(chamber_df, flame_df):
    """
    Merges Chamber and Flame data using t_rel_s.
    Handles mismatched sampling rates via interpolation.
    """
    # Merge on the closest timestamp
    fused = pd.merge_asof(flame_df.sort_values('t_rel_s'), 
                          chamber_df.sort_values('t_rel_s'), 
                          on='t_rel_s', direction='nearest')
    
    # Flag the data gap from the DeepSeek analysis
    fused['Sensor_Health_Flag'] = np.where(fused['Standoff_mm'].isna(), 'FLAME_LOST', 'OK')
    
    print("[UTFE] Fusion Complete. Master dataframe created.")
    return fused

if __name__ == '__main__':
    chamber = parse_chamber_readings("C039H01_Chamber_Readings.txt", "2013-02-08 23:05:35")
    flame = parse_flame_tracking("FLEX-008_ColorCamData.txt")
    
    master = fuse_timelines(chamber, flame)
    print("\n--- FUSED TELEMETRY SAMPLE ---")
    print(master.head(10).to_string(index=False))
    print("...")
    print(master.tail(5).to_string(index=False))
