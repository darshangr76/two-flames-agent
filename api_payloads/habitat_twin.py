"""
ZENITHFORGE - Habitat Digital Twin
Runs a "rehearsal" of all 6 materials against a target habitat before the engineer commits.
"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from api_payloads.query_engine import inverse_query, predict_loc_bounded, adjust_for_gravity

HABITATS = {
    "lunar":  {"o2": 30.0, "pressure": 56.5, "flow": 5.0, "gravity": 0.16},
    "mars":   {"o2": 21.0, "pressure": 101.3, "flow": 2.0, "gravity": 0.38},
    "rover":  {"o2": 30.0, "pressure": 70.0, "flow": 8.0, "gravity": 0.38},
}
MATERIALS = ["NOMEX", "PMMA", "n-heptane", "n-octane", "n-decane", "SIBAL fabric"]

def rehearse_habitat(habitat_name):
    spec = HABITATS.get(habitat_name.lower())
    if not spec:
        raise ValueError(f"Unknown habitat: {habitat_name}")
    results = []
    for mat in MATERIALS:
        q = inverse_query(mat, spec["o2"], spec["pressure"], spec["flow"], habitat_name)
        if q.get("status") != "SUCCESS":
            results.append({"material": mat, "status": "NO_DATA"})
            continue
        pred = predict_loc_bounded(mat, spec["o2"], spec["pressure"], spec["flow"], habitat_name)
        gravity_loc = None
        if pred.get("status") == "SUCCESS":
            gravity_loc = adjust_for_gravity(pred["predicted_LOC_pct"], spec["gravity"])
        results.append({
            "material": mat,
            "confidence": q["confidence"],
            "warning": q.get("warning", ""),
            "bounded_LOC_microg": pred.get("predicted_LOC_pct"),
            "LOC_at_habitat_gravity": gravity_loc,
            "top_experiment": q["top_matches"][0]["experiment_id"] if q["top_matches"] else None,
        })
    return {"habitat": habitat_name, "spec": spec, "materials": results}

if __name__ == '__main__':
    for h in ["lunar", "mars", "rover"]:
        print(f"\n=== REHEARSAL: {h.upper()} HABITAT ===")
        result = rehearse_habitat(h)
        for r in result["materials"]:
            print(f"  {r['material']:15s} | conf={r['confidence']:6s} | LOC_microg={r.get('bounded_LOC_microg')} | LOC_habitat={r.get('LOC_at_habitat_gravity')}")
