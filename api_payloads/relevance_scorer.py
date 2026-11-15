"""
ZENITHFORGE - Mission Relevance Scorer
Ranks investigations by match to a target habitat spec, not just numerical distance.
"""
import pandas as pd
import numpy as np
import os

MASTER = os.path.join(os.path.dirname(__file__), '..', 'clean_data', 'zenithforge_master.csv')

HABITAT_SPECS = {
    "lunar":  {"o2": 30.0, "pressure": 56.5, "flow": 5.0,  "gravity": 0.16},
    "mars":   {"o2": 21.0, "pressure": 101.3, "flow": 2.0, "gravity": 0.38},
    "rover":  {"o2": 30.0, "pressure": 70.0, "flow": 8.0,  "gravity": 0.38},
    "iss":    {"o2": 21.0, "pressure": 101.3, "flow": 3.0, "gravity": 0.0},
}

def _recency_weight(inv_id):
    if not isinstance(inv_id, str): return 0.5
    u = inv_id.upper()
    if 'FLARE' in u: return 1.0
    if 'SAFFIRE' in u: return 0.9
    if 'FLEX' in u or 'CFI' in u: return 0.7
    if 'BASS' in u: return 0.6
    return 0.5

def mission_relevance(target_mission, material):
    spec = HABITAT_SPECS.get(target_mission.lower())
    if spec is None:
        raise ValueError(f"Unknown mission '{target_mission}'. Use: {list(HABITAT_SPECS)}")
    df = pd.read_csv(MASTER)
    df = df[df['material'].str.contains(material, case=False, na=False)].copy()
    if df.empty:
        return df
    df['atmo_score'] = 1.0 - np.minimum(1.0,
        (np.abs(df['oxygen_concentration_pct'] - spec['o2']) / 50.0) * 0.5 +
        (np.abs(df['pressure_kpa'] - spec['pressure']) / 100.0) * 0.5)
    df['material_score'] = 1.0
    df['gravity_score'] = 1.0
    df['recency_score'] = df['investigation_id'].apply(_recency_weight)
    qmap = {'A': 1.0, 'B': 0.6, 'C': 0.2}
    df['quality_score'] = df['data_quality'].map(qmap).fillna(0.3)
    df['mission_relevance'] = (0.30*df['atmo_score'] + 0.25*df['material_score'] +
                               0.20*df['gravity_score'] + 0.15*df['recency_score'] +
                               0.10*df['quality_score'])
    return df.sort_values('mission_relevance', ascending=False)

if __name__ == '__main__':
    print("=== NOMEX on Lunar ===")
    r = mission_relevance("lunar", "NOMEX")
    print(r[['experiment_id','flame_regime','data_quality','mission_relevance','atmo_score','recency_score']].head(5).to_string(index=False))
    print("\n=== PMMA on Mars ===")
    r = mission_relevance("mars", "PMMA")
    print(r[['experiment_id','flame_regime','data_quality','mission_relevance','atmo_score','recency_score']].head(5).to_string(index=False))
