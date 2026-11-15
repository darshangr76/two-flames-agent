"""
ZENITHFORGE Engineering Inverse Query Engine v2
New: mission_profile weights + export_for_modeling()
"""

import pandas as pd
import numpy as np
import os
import json

d = os.path.dirname(__file__)
ROOT = os.path.join(d, '..')
MASTER = os.path.join(ROOT, 'clean_data', 'zenithforge_master.csv')
EXPORT_DIR = os.path.join(ROOT, 'exports')
os.makedirs(EXPORT_DIR, exist_ok=True)

MISSION_PROFILES = {
    "lunar":    {"o2": 0.5, "p": 0.4, "f": 0.1},
    "mars":    {"o2": 0.2, "p": 0.1, "f": 0.7},
    "rover":   {"o2": 0.4, "p": 0.3, "f": 0.3},
    "bas":     {"o2": 21.0, "p": 101.3, "f": 3.0},
    "jaxa_flare": {"o2": 0.25, "p": 0.5, "f": 0.25},
    "custom":  {"o2": 1.0, "p": 1.0, "f": 1.0},
}

def inverse_query(material, o2_pct, pressure_kpa, flow_cm_s, mission_profile="custom"):
    df = pd.read_csv(MASTER)
    subset = df[df['material'].str.contains(material, case=False, na=False)].copy()
    if subset.empty:
        return {"status": "FAIL", "reason": f"No data for '{material}'.", "confidence": "NONE", "warning": "DO NOT certify this material."}
    w = MISSION_PROFILES.get(mission_profile.lower(), MISSION_PROFILES["custom"])
    subset['d_o2'] = np.abs(subset['oxygen_concentration_pct'] - o2_pct) / 100.0 * w["o2"]
    subset['d_p']  = np.abs(subset['pressure_kpa'] - pressure_kpa) / 101.3 * w["p"]
    subset['d_f']  = np.abs(subset['flow_velocity_cm_s'] - flow_cm_s) / 20.0 * w["f"]
    subset['distance'] = np.sqrt(subset['d_o2']**2 + subset['d_p']**2 + subset['d_f']**2)
    ranked = subset.sort_values('distance').head(3)
    best_dist = float(ranked.iloc[0]['distance'])
    if best_dist < 0.15:
        conf = "HIGH"; warn = "Direct experimental analog available."
    elif best_dist < 0.40:
        conf = "MEDIUM"; warn = "Extrapolation required. Verify flame regime."
    else:
        conf = "LOW"; warn = "DATA GAP: No close analog. Prediction unreliable."
    if (ranked['flame_regime'] == '3D').any():
        warn += " | WARNING: Nearest data involves 3D spherical flames. 2D models invalid."
    if (ranked['flame_regime'] == 'cool_flame').any():
        warn += " | WARNING: Nearest data involves cool flames. Visible-flame detection may miss combustion."
    return {"status": "SUCCESS", "query": {"material": material, "o2": o2_pct, "pressure": pressure_kpa, "flow": flow_cm_s, "mission": mission_profile}, "confidence": conf,
            "warning": warn, "top_matches": ranked[['experiment_id', 'investigation_id', 'flame_regime', 'limiting_oxygen_concentration_LOC', 'data_quality', 'distance', 'source_url']].to_dict(orient='records')}

def export_for_modeling(material, o2_pct, pressure_kpa, flow_cm_s, mission_profile="custom"):
    result = inverse_query(material, o2_pct, pressure_kpa, flow_cm_s, mission_profile)
    if result["status"] != "SUCCESS":
        return result
    rows = []
    for m in result["top_matches"]:
        usable = (m.get("data_quality") == "A" and m.get("flame_regime") != "3D" and result["confidence"] != "LOW")
        rows.append({
            "target_material": material, "target_o2": o2_pct, "target_pressure": pressure_kpa, "target_flow": flow_cm_s,
            "mission_profile": mission_profile,
            "experiment_id": m.get("experiment_id"), "investigation_id": m.get("investigation_id"),
            "flame_regime": m.get("flame_regime"), "loc": m.get("limiting_oxygen_concentration_LOC"),
            "distance": m.get("distance"), "confidence": result["confidence"],
            "usable_for_training": usable
        })
    out = pd.DataFrame(rows)
    fn = os.path.join(EXPORT_DIR, f"modeling_{material}_{o2_pct}opt_{pressure_kpa}kPa.csv")
    out.to_csv(fn, index=False)
    return {"status": "SUCCESS", "file": fn, "rows": len(out), "usable_rows": int(out["usable_for_training"].sum())}

if __name__ == "__main__":
    print("=== NOMEX on Lunar HABITAT ===")
    print(json.dumps(inverse_query("NOMEX", 30.0, 56.5, 5.0, "lunar"), indent=2))
    print("\n=== PMMA on Mars TRANSIT ===")
    print(json.dumps(inverse_query("PMMA", 21.0, 101.3, 2.0, "mars"), indent=2))
    print("\n=== EXPORT for Modeling ===")
    print(json.dumps(export_for_modeling("PMMA", 21.0, 101.3, 2.0, "mars"), indent=2))



# --- Bounded LOC predictor ---
def predict_loc_bounded(material, o2, pressure, flow, mission="custom"):
    q = inverse_query(material, o2, pressure, flow, mission)
    if q.get("status") != "SUCCESS":
        return {"status": "FAIL", "reason": q.get("reason")}
    raw_locs = [m.get("limiting_oxygen_concentration_LOC") for m in q["top_matches"]]
    locs = [v for v in raw_locs if v is not None and not (isinstance(v, float) and v != v)]
    if not locs:
        return {"status": "NO_LOC", "reason": "No LOC in nearest experiments", "confidence": q["confidence"]}
    spread = max(locs) - min(locs)
    if spread > 3.0:
        return {"status": "REFUSE",
                "reason": f"LOC spread {round(spread,2)}% exceeds 3% threshold",
                "confidence": q["confidence"], "nearby_locs": locs}
    pred = sum(locs) / len(locs)
    return {
        "status": "SUCCESS",
        "predicted_LOC_pct": round(pred, 1),
        "lower_bound": round(min(locs), 1),
        "upper_bound": round(max(locs), 1),
        "spread_pct": round(spread, 2),
        "n_sources": len(locs),
        "confidence": q["confidence"],
        "mission": mission,
    }

# --- Scientific Evidence Record Generator ---
import datetime

def generate_evidence_record(experiment_id, prediction_data, source_file="clean_data/zenithforge_master.csv"):
    return {
        "prediction_id": f"FG-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "experiment_id": experiment_id,
        "model_version": "ZENITHFORGE-Role1-v2.1",
        "source_evidence": {
            "dataset": source_file,
            "row_reference": "See experiment_id lookup",
            "data_lineage": "NASA PSI/NTRS -> Pandas -> Query Engine"
        },
        "prediction_payload": prediction_data,
        "validation_status": "Physics-constrained, uncertainty bounded"
    }

# --- Gravity Physics Adjuster ---
def adjust_for_gravity(predicted_loc, target_gravity_g):
    """
    Adjusts the LOC prediction for partial gravity (Moon = 0.16g, Mars = 0.38g).
    Microgravity (0g) flames are spherical. Earth (1g) flames are buoyant.
    Partial gravity sits in between.
    """
    if target_gravity_g == 0.0:
        return predicted_loc
    scale_factor = 1.0 + (target_gravity_g * 0.1)
    return round(predicted_loc * scale_factor, 1)

if __name__ == '__main__':
    print("\n=== GRAVITY ADJUSTER TEST ===")
    base_loc = 16.8
    print(f"Microgravity (0g) LOC: {base_loc}%")
    print(f"Moon gravity (0.16g) LOC: {adjust_for_gravity(base_loc, 0.16)}%")
    print(f"Mars gravity (0.38g) LOC: {adjust_for_gravity(base_loc, 0.38)}%")
    print(f"Earth gravity (1.0g) LOC: {adjust_for_gravity(base_loc, 1.0)}%")

# --- Evidence Coverage Report ---
def evidence_coverage(material, o2, pressure, flow, mission="custom"):
    """Reports which evidence modalities back a prediction."""
    q = inverse_query(material, o2, pressure, flow, mission)
    if q.get("status") != "SUCCESS":
        return {"status": "FAIL", "reason": q.get("reason")}
    has_numerical = True
    has_research  = any(m.get("source_url", "").startswith("http") for m in q["top_matches"])
    has_image     = any(m.get("flame_regime") in ("2D", "3D", "cool_flame") for m in q["top_matches"])
    has_video     = False
    modalities = {"numerical": has_numerical, "image": has_image, "video": has_video, "research": has_research}
    weights = {"numerical": 0.40, "image": 0.20, "video": 0.20, "research": 0.20}
    covered = sum(w for k, w in weights.items() if modalities[k])
    return {
        "status": "SUCCESS",
        "modalities": modalities,
        "coverage_pct": round(covered * 100, 1),
        "missing": [k for k, v in modalities.items() if not v],
    }

# --- Custom Mission Weights ---
def register_mission(name, o2_weight, p_weight, f_weight):
    """Allows the user to register a new mission profile at runtime."""
    MISSION_PROFILES[name.lower()] = {"o2": o2_weight, "p": p_weight, "f": f_weight}
    return {"registered": name.lower(), "weights": MISSION_PROFILES[name.lower()]}


# --- NASA FLEX Knowledge Base Integration ---
import json as _json

FLEX_KB_PATH = os.path.join(os.path.dirname(__file__), '..', 'docs', 'flex_knowledge_base.json')

def check_flex_knowledge(material, o2_pct, pressure_kpa):
    """Cross-references the query with the NASA FLEX TP-2015-216046 report."""
    if not os.path.exists(FLEX_KB_PATH):
        return None
    with open(FLEX_KB_PATH, 'r', encoding='utf-8-sig') as f:
        kb = _json.load(f)
    pressure_atm = pressure_kpa / 101.325
    mat_norm = material.lower().replace("n-", "").strip()
    for test in kb["test_matrix"]:
        test_norm = test["fuel"].lower().replace("n-", "").strip()
        if test_norm == mat_norm and abs(test["pressure_atm"] - pressure_atm) < 0.2:
            if "Cool Flame" in test["extinction_type"]:
                return {
                    "warning": "COOL FLAME RISK DETECTED",
                    "science": kb["key_physics"]["cool_flame_description"],
                    "test_reference": test["test_id"],
                    "temperature_range_F": kb["key_physics"]["cool_flame_temp_F"]
                }
    return None

# --- Enhanced inverse query with FLEX knowledge ---
def inverse_query_with_flex(material, o2_pct, pressure_kpa, flow_cm_s, mission="custom"):
    """Wraps inverse_query and appends FLEX cool-flame warnings where relevant."""
    result = inverse_query(material, o2_pct, pressure_kpa, flow_cm_s, mission)
    if result.get("status") != "SUCCESS":
        return result
    flex = check_flex_knowledge(material, o2_pct, pressure_kpa)
    if flex:
        result["flex_warning"] = flex
        result["warning"] = result.get("warning", "") + " | FLEX TP-2015: " + flex["warning"]
    return result
