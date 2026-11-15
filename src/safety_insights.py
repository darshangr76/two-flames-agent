"""
ZENITHFORGE - Spacecraft Safety Insights
Translates NASA FLEX numerical findings into engineering recommendations.
"""

def generate_safety_insight(material, pressure_atm, o2_pct, flame_regime):
    """Maps data back to spacecraft fire safety design."""
    insights = []
    if material.lower() in ["nomex", "sibal fabric"] and flame_regime == "3D":
        insights.append("MATERIAL RISK: 3D spherical flame detected. NOMEX may burn more vigorously than 2D models predict in low-flow environments.")
    if o2_pct > 25.0 and pressure_atm > 1.5:
        insights.append("ATMOSPHERE RISK: High O2 + High Pressure environment. Extinction limits are dangerously high. Disruption likely.")
    elif o2_pct < 15.0:
        insights.append("ATMOSPHERE INSIGHT: Low O2 environment. Fire is likely to self-extinguish, but may smolder.")
    if flame_regime == "cool_flame":
        insights.append("DETECTION WARNING: Cool flame regime confirmed. Visible-light sensors will fail. Recommend IR or thermal detection.")
    if not insights:
        insights.append("NOMINAL: Nominal conditions. Standard fire safety protocols apply.")
    return insights

if __name__ == '__main__':
    print("--- SPACECRAFT SAFETY INSIGHTS ---")
    print("Scenario 1: NOMEX at 30% O2, 0.5 atm (Lunar Habitat)")
    for i in generate_safety_insight("NOMEX", 0.5, 30.0, "3D"):
        print(f"  -> {i}")
    print("\nScenario 2: Heptane at 21% O2, 1 atm (ISS)")
    for i in generate_safety_insight("Heptane", 1.0, 21.0, "cool_flame"):
        print(f"  -> {i}")
