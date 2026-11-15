"""
ZENITHFORGE - Evidence Fusion Engine
Combines multi-modal evidence into a single trust score.
"""
import pandas as pd

WEIGHTS = {
    "numerical": 0.40,
    "image": 0.20,
    "video": 0.20,
    "research": 0.20
}

def fuse_evidence(evidence_dict):
    """
    evidence_dict format: 
    {
      "experiment_id": "FLARE-NOMEX-11",
      "numerical_score": 0.95, "numerical_available": True,
      "image_score": 0.80,     "image_available": True,
      "video_score": None,     "video_available": False,
      "research_score": 0.90,  "research_available": True
    }
    """
    weighted_sum = 0.0
    available_weight = 0.0
    modalities_used = 0
    
    for mod, weight in WEIGHTS.items():
        score = evidence_dict.get(f"{mod}_score")
        available = evidence_dict.get(f"{mod}_available", False)
        
        if available and score is not None:
            weighted_sum += (score * weight)
            available_weight += weight
            modalities_used += 1
            
    if available_weight == 0:
        return {"experiment_id": evidence_dict.get("experiment_id"), "trust_score": 0, "coverage": "NONE"}
        
    final_score = (weighted_sum / available_weight) * 100
    
    return {
        "experiment_id": evidence_dict.get("experiment_id"),
        "trust_score": round(final_score, 1),
        "modalities_used": modalities_used,
        "coverage_percentage": round(available_weight * 100, 0)
    }

if __name__ == '__main__':
    # Test Case 1: All modalities available
    test1 = {
        "experiment_id": "FLARE-NOMEX-11",
        "numerical_score": 0.95, "numerical_available": True,
        "image_score": 0.85, "image_available": True,
        "video_score": 0.90, "video_available": True,
        "research_score": 0.80, "research_available": True
    }
    print("Test 1 (Full evidence):", fuse_evidence(test1))
    
    # Test Case 2: Missing video (should normalize weight)
    test2 = {
        "experiment_id": "BASS2-142",
        "numerical_score": 0.90, "numerical_available": True,
        "image_score": 0.70, "image_available": True,
        "video_score": None, "video_available": False,
        "research_score": 0.95, "research_available": True
    }
    print("Test 2 (Missing video):", fuse_evidence(test2))
