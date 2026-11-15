"""
ZENITHFORGE - Coverage Matrix
Material x O2-band x pressure-band coverage grid -> docs/coverage_matrix.csv
"""
import pandas as pd
import os, itertools

MASTER = os.path.join(os.path.dirname(__file__), '..', 'clean_data', 'zenithforge_master.csv')
OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'coverage_matrix.csv')

def build():
    df = pd.read_csv(MASTER)
    materials = sorted(df['material'].dropna().unique())
    o2_bands = [(0,18),(18,25),(25,35),(35,100)]
    p_bands = [(0,50),(50,90),(90,110),(110,10000)]
    rows = []
    for mat, o2, p in itertools.product(materials, o2_bands, p_bands):
        s = df[(df['material']==mat) &
               (df['oxygen_concentration_pct']>=o2[0]) &
               (df['oxygen_concentration_pct']<o2[1]) &
               (df['pressure_kpa']>=p[0]) &
               (df['pressure_kpa']<p[1])]
        n = len(s)
        rows.append({
            'material': mat,
            'o2_band': f"{o2[0]}-{o2[1]}",
            'pressure_band': f"{p[0]}-{p[1]}_kPa",
            'experiment_count': n,
            'regimes_present': ','.join(sorted(s['flame_regime'].dropna().unique())) if n else '',
            'coverage_status': 'COVERED' if n >= 2 else ('THIN' if n == 1 else 'GAP'),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"[COVERAGE] {len(out)} combos -> {OUT}")
    print(out['coverage_status'].value_counts().to_string())
    return out

if __name__ == '__main__':
    build()
