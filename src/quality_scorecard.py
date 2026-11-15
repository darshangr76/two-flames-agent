"""
ZENITHFORGE - Sensor Health & Data-Quality Scorecard
"""
import pandas as pd
import numpy as np

def calculate_quality_score(df, run_name):
    score = 100
    issues = []
    missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 5:
        score -= (missing_pct * 1.5)
        issues.append(f"High missing data density ({missing_pct:.1f}%)")
    if 'Igniter_Voltage_V' in df.columns:
        neg_v = df[df['Igniter_Voltage_V'] < 0]
        if not neg_v.empty:
            score -= 20
            issues.append(f"Negative igniter voltage detected ({len(neg_v)} samples)")
    if 't_rel_s' in df.columns:
        if not df['t_rel_s'].is_monotonic_increasing:
            score -= 15
            issues.append("Non-monotonic timestamps detected")
    if 'Standoff_mm' in df.columns:
        gaps = df['Standoff_mm'].isna().sum()
        if gaps > 0:
            score -= 10
            issues.append(f"Flame tracking lost ({gaps} frames)")
    final_score = max(0, score)
    return {
        "run_name": run_name,
        "quality_score": round(final_score, 1),
        "status": "PASS" if final_score > 80 else "WARN" if final_score > 50 else "FAIL",
        "issues_found": issues
    }

if __name__ == '__main__':
    faulty_data = pd.DataFrame({
        "t_rel_s": [0, 1, 3, 2, 4],
        "Igniter_Voltage_V": [1.2, 1.2, -0.01, 1.2, 1.2],
        "Standoff_mm": [8.0, 7.5, np.nan, np.nan, np.nan]
    })
    report = calculate_quality_score(faulty_data, "C039H03_Audit")
    print("--- SENSOR HEALTH SCORECARD ---")
    print(f"Run: {report['run_name']}")
    print(f"Score: {report['quality_score']}/100 ({report['status']})")
    print("Issues Detected:")
    for issue in report['issues_found']:
        print(f"  - {issue}")
