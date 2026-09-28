"""Context coverage of frozen inspection orders; no learned outputs are recomputed."""
from pathlib import Path
from collections import Counter, defaultdict
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
    paths = {
        'protocol': ROOT / 'experiments/protocols/inspection_context_coverage_r1.json',
        'catalog': ROOT / 'solution/evidence_intake/demo/collection_catalog.json',
        'census': base / 'region_family_census_r2/family_census.csv',
        'orders': base / 'inspection_planning_r1/selected_photos.csv',
    }
    catalog = json.loads(paths['catalog'].read_text(encoding='utf-8'))
    photos = photo_index(catalog)
    census = pd.read_csv(paths['census']).set_index('family_internal')
    orders = pd.read_csv(paths['orders'])
    ids = sorted(photos)
    runs = [(method, 0, group.sort_values('rank').canonical_id.tolist())
            for method, group in orders.groupby('method')]
    runs += [('random', seed, np.random.default_rng(seed).permutation(ids).tolist()) for seed in range(100)]
    out = base / 'inspection_context_coverage_r1'
    out.mkdir(exist_ok=True)
    metrics, details = [], []
    stable = set(census.index[census.partition_jaccard_median >= .5])
    for method, seed, order in runs:
        for budget in [10, 25, 50, 100]:
            counts = defaultdict(Counter)
            for parent in order[:budget]:
                for family in photos[parent]['families']:
                    counts[family][photos[parent]['global_community']] += 1
            cross = {f for f, cc in counts.items() if len(cc) >= 2}
            repeated = {f for f, cc in counts.items() if sum(n >= 2 for n in cc.values()) >= 2}
            metrics.append(dict(method=method, replicate=seed, budget=budget,
                                cross_context_families=len(cross),
                                cross_context_repeated=len(repeated),
                                stable_cross_context_families=len(cross & stable)))
            if method != 'random' and budget == 50:
                for family in census.index:
                    cc = counts[family]
                    row = census.loc[family]
                    details.append(dict(method=method, family_internal=int(family),
                                        visual_name=row.visual_name_id,
                                        selected_photos=sum(cc.values()), selected_contexts=len(cc),
                                        contexts_with_two_photos=sum(n >= 2 for n in cc.values()),
                                        counts=json.dumps(dict(cc), sort_keys=True),
                                        full_contexts=int(row.global_community_span),
                                        median_partition_jaccard=row.partition_jaccard_median,
                                        coherence=row.coherence_0_4, shortcut_risk=row.shortcut_risk_0_4,
                                        semantic_status=row.semantic_status))
    df = pd.DataFrame(metrics)
    df.to_csv(out / 'evaluation.csv', index=False)
    pd.DataFrame(details).to_csv(out / 'family_coverage_at_50.csv', index=False)
    # Fix one inspectable pair per discovered bridge without using semantic ratings.
    selected = orders[(orders.method == 'discovery_greedy') & (orders['rank'] <= 50)]
    ranks = dict(zip(selected.canonical_id, selected['rank']))
    bridge_families = sorted(r['family_internal'] for r in details
                             if r['method'] == 'discovery_greedy' and r['selected_contexts'] >= 2)
    pair_items = []
    for family in bridge_families:
        regions = sorted((r for r in catalog['items'] if r['family_internal'] == family and r['parent_id'] in ranks),
                         key=lambda r: (ranks[r['parent_id']], -r['family_affinity']))
        communities = set()
        for region in regions:
            if region['global_community'] not in communities:
                pair_items.append(region)
                communities.add(region['global_community'])
            if len(communities) == 2:
                break
    (out / 'pair_review_manifest.json').write_text(json.dumps(pair_items, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {
        'inputs': {key: {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in paths.items()},
        'eligible_photos': len(photos),
        'families': len(census),
        'full_collection_cross_context_families': int((census.global_community_span >= 2).sum()),
        'claim': 'Frozen r2 memberships across global communities; this is not a count of independently verified semantic bridges.',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(df.groupby(['method', 'budget']).mean(numeric_only=True).drop(columns='replicate').to_string())
    print(pd.DataFrame(details).query("method == 'discovery_greedy' and selected_contexts >= 2").to_string(index=False))


if __name__ == '__main__':
    main()
