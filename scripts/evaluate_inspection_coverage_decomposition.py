"""Decompose frozen inspection coverage without changing selection orders."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution.evidence_intake.planner import photo_index


def main():
    base = ROOT / 'experiments/final_study_20260928'
    inputs = {
        'protocol': ROOT / 'experiments/protocols/inspection_coverage_decomposition_r1.json',
        'catalog': ROOT / 'solution/evidence_intake/demo/collection_catalog.json',
        'census': base / 'region_family_census_r2/family_census.csv',
        'orders': base / 'inspection_planning_r1/selected_photos.csv',
    }
    protocol = json.loads(inputs['protocol'].read_text())
    photos = photo_index(json.loads(inputs['catalog'].read_text()))
    census = pd.read_csv(inputs['census']).set_index('family_internal')
    orders = pd.read_csv(inputs['orders'])
    all_families = set(census.index)
    assert len(all_families) == 129
    strata = {
        'all': all_families,
        'stable': set(census.index[census.partition_jaccard_median >= .5]),
        'rare_2_4_photos': set(census.index[census.unique_parent_photos.between(2, 4)]),
        'at_least_5_photos': set(census.index[census.unique_parent_photos >= 5]),
    }
    for name in ['rare_2_4_photos', 'at_least_5_photos']:
        strata['stable_' + name] = strata['stable'] & strata[name]
    for risk in range(5):
        strata[f'shortcut_risk_{risk}'] = set(census.index[census.shortcut_risk_0_4 == risk])
    assert strata['rare_2_4_photos'] | strata['at_least_5_photos'] == all_families
    assert not strata['rare_2_4_photos'] & strata['at_least_5_photos']
    runs = [(name, 0, g.sort_values('rank').canonical_id.tolist())
            for name, g in orders.groupby('method')]
    runs += [('random', seed, np.random.default_rng(seed).permutation(sorted(photos)).tolist())
             for seed in protocol['random_seeds']]
    rows, family_rows = [], []
    for method, seed, order in runs:
        assert len(order) == len(set(order))
        for budget in protocol['budgets']:
            counts = Counter()
            contexts = defaultdict(set)
            for parent in order[:budget]:
                for family in photos[parent]['families']:
                    counts[family] += 1
                    contexts[family].add(photos[parent]['global_community'])
            seen = {f for f, n in counts.items() if n > 0}
            repeated = {f for f, n in counts.items() if n >= 2}
            cross = {f for f, cc in contexts.items() if len(cc) >= 2}
            for stratum, members in strata.items():
                rows.append(dict(method=method, replicate=seed, budget=budget,
                                 stratum=stratum, eligible_families=len(members),
                                 seen=len(seen & members), repeated=len(repeated & members),
                                 cross_context=len(cross & members)))
            if method != 'random' and budget == 50:
                for family, c in census.iterrows():
                    family_rows.append(dict(method=method, family_internal=int(family),
                                            photo_count=counts[family], context_count=len(contexts[family]),
                                            stable=family in strata['stable'],
                                            parent_photo_count=int(c.unique_parent_photos),
                                            shortcut_risk=int(c.shortcut_risk_0_4)))
    df = pd.DataFrame(rows)
    known = pd.read_csv(base / 'inspection_context_coverage_r1/evaluation.csv')
    computed = df.query("stratum == 'stable'").merge(known, on=['method', 'replicate', 'budget'])
    assert (computed.cross_context == computed.stable_cross_context_families).all()
    at50 = df.query('budget == 50')
    summary_rows = []
    for (method, stratum), group in at50.groupby(['method', 'stratum']):
        summary_rows.append(dict(method=method, stratum=stratum,
                                 eligible_families=int(group.eligible_families.iloc[0]),
                                 seen_mean=float(group.seen.mean()),
                                 seen_p025=float(group.seen.quantile(.025)),
                                 seen_p975=float(group.seen.quantile(.975)),
                                 repeated_mean=float(group.repeated.mean()),
                                 cross_context_mean=float(group.cross_context.mean())))
    out = base / 'inspection_coverage_decomposition_r1'
    out.mkdir(exist_ok=True)
    df.to_csv(out / 'evaluation.csv', index=False)
    pd.DataFrame(family_rows).to_csv(out / 'all_family_coverage_at_50.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / 'summary_at_50.csv', index=False)
    receipt = {
        'inputs': {k: {'path': p.relative_to(ROOT).as_posix(),
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                   for k, p in inputs.items()},
        'eligible_photos': len(photos), 'families': len(census),
        'strata_counts': {k: len(v) for k, v in strata.items()},
        'existing_context_evaluation_matches': len(computed),
        'design': protocol['design'], 'interpretation': protocol['interpretation'],
        'random_intervals': 'Empirical central 95% of 100 random orders, not an independent-sample confidence interval.',
    }
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(summary.query("stratum in ['all','stable','rare_2_4_photos','at_least_5_photos','stable_rare_2_4_photos','stable_at_least_5_photos']").to_string(index=False))


if __name__ == '__main__':
    main()
