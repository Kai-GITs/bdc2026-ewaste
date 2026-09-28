"""Compare instance selection on a frozen batch without semantic-label inputs."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    base = ROOT / 'experiments/final_study_20260928'
    local = ROOT / '.local/final_study_20260928'
    paths = {
        'protocol': ROOT / 'experiments/protocols/context_pair_selection_r1.json',
        'catalog': ROOT / 'solution/evidence_intake/demo/collection_catalog.json',
        'orders': base / 'inspection_planning_r1/selected_photos.csv',
        'baseline': base / 'inspection_context_coverage_r1/pair_review_manifest.json',
        'assignments': local / 'region-family-graph-object-filtered-r2/assignments.csv.gz',
        'foreground': local / 'composition-region-bank-r1/dinov3_foreground.npy',
        'box': local / 'dinov3-region-box-r1/dinov3_box.npy',
        'siglip': local / 'siglip2-region-crop-r1/siglip2_crop.npy',
    }
    orders = pd.read_csv(paths['orders'])
    parents = set(orders.loc[(orders.method == 'discovery_greedy') & (orders['rank'] <= 50), 'canonical_id'])
    assert len(parents) == 50
    baseline = json.loads(paths['baseline'].read_text(encoding='utf-8'))
    families = sorted({r['family_internal'] for r in baseline})
    assert len(families) == 25 and len(baseline) == 50
    items = sorted([r for r in json.loads(paths['catalog'].read_text(encoding='utf-8'))['items']
                    if r['parent_id'] in parents and r['family_internal'] in families], key=lambda r: r['region_id'])
    assignment = pd.read_csv(paths['assignments']).set_index('region_id')
    rows = np.array([int(assignment.loc[r['region_id'], 'feature_row']) for r in items])
    vectors = []
    for name in ['foreground', 'box', 'siglip']:
        x = np.asarray(np.load(paths[name], mmap_mode='r')[rows], dtype=np.float32)
        assert np.isfinite(x).all()
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        vectors.append(x)
    by_id = {r['region_id']: r for r in items}
    selected, candidates = [], []
    for family in families:
        members = [i for i, r in enumerate(items) if r['family_internal'] == family]
        pairs = [(i, j) for i, j in combinations(members, 2)
                 if items[i]['parent_id'] != items[j]['parent_id']
                 and items[i]['source_sha256'] != items[j]['source_sha256']
                 and items[i]['global_community'] != items[j]['global_community']]
        assert pairs, family
        left, right = np.array(pairs).T
        scores = np.column_stack([np.einsum('ij,ij->i', x[left], x[right]) for x in vectors])
        ranks = np.column_stack([rankdata(scores[:, k], method='average') / len(pairs) for k in range(3)])
        records = []
        for k, (i, j) in enumerate(pairs):
            records.append(dict(family_internal=family, left_id=items[i]['region_id'], right_id=items[j]['region_id'],
                                foreground=float(scores[k, 0]), box=float(scores[k, 1]), siglip=float(scores[k, 2]),
                                min_rank=float(ranks[k].min()), mean_rank=float(ranks[k].mean()), eligible_pairs=len(pairs)))
        old_ids = {r['region_id'] for r in baseline if r['family_internal'] == family}
        old = next(r for r in records if {r['left_id'], r['right_id']} == old_ids)
        fg = sorted(records, key=lambda r: (-r['foreground'], r['left_id'], r['right_id']))[0]
        mv = sorted(records, key=lambda r: (-r['min_rank'], -r['mean_rank'], -r['foreground'], r['left_id'], r['right_id']))[0]
        for method, record in [('first_encounter', old), ('foreground_nearest', fg), ('view_consensus', mv)]:
            pair_id = hashlib.sha256((record['left_id'] + '|' + record['right_id']).encode()).hexdigest()[:12]
            selected.append(dict(method=method, pair_id=pair_id, **record))
        candidates.extend(records)
    out = base / 'context_pair_selection_r1'
    out.mkdir(exist_ok=True)
    pd.DataFrame(selected).to_csv(out / 'selected_pairs.csv', index=False)
    pd.DataFrame(candidates).to_csv(out / 'eligible_pair_scores.csv', index=False)
    unique = {r['pair_id']: r for r in selected}
    # Shuffle display order deterministically; no family labels or scores in sheets.
    ids = sorted(unique)
    np.random.default_rng(260928).shuffle(ids)
    review = [dict(review_id=f'P{i+1:02d}', pair_id=pid,
                   left=by_id[unique[pid]['left_id']], right=by_id[unique[pid]['right_id']]) for i, pid in enumerate(ids)]
    (out / 'review_manifest.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = dict(inputs={k: {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for k, p in paths.items()},
                   families=len(families), selected_photos=len(parents), selected_feature_rows=len(rows),
                   eligible_pairs=len(candidates), unique_review_pairs=len(review),
                   claim_scope='Adaptive transductive mechanism experiment; no semantic result until original-image inspection.')
    (out / 'selection_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'inputs'}))

if __name__ == '__main__':
    main()
