"""Post-hoc community control using frozen predictions, without refitting.

Report every whole-image community represented in the 48-photo relation cohort.
The AUC is estimable only where both outcomes are present. Paired stratified
bootstrap resamples identical parent photos for both representation scores.
This controls coarse community membership, not website source or all context.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]


def run():
    pred = P / 'experiments/final_study_20260928/exposure_state_holdout_r1/blind_test_predictions.csv'
    membership = P / 'experiments/multiscale_graph_20260928/global/canonical_assignments.csv'
    out = P / 'experiments/final_study_20260928/exposure_state_community_control_r1'
    out.mkdir(exist_ok=True)
    df = pd.read_csv(pred).merge(pd.read_csv(membership)[['canonical_id', 'community_leiden_fused']], on='canonical_id', validate='one_to_one')
    assert len(df) == 48
    rng = np.random.default_rng(20260928)
    results = []
    details = []
    for cid, group in df.groupby('community_leiden_fused', sort=True):
        pos = group[group.target == 1]
        neg = group[group.target == 0]
        rec = {'community': int(cid), 'n': len(group), 'mounted': len(pos), 'detached_or_multiple': len(neg), 'estimable': bool(len(pos) and len(neg))}
        if not rec['estimable']:
            results.append(rec)
            continue
        matrices = {}
        ip = rng.integers(0, len(pos), size=(10000, len(pos)))
        jn = rng.integers(0, len(neg), size=(10000, len(neg)))
        boot = {}
        for column in [c for c in df.columns if c.startswith('score_')]:
            a = pos[column].to_numpy()[:, None]
            b = neg[column].to_numpy()[None, :]
            matrix = (a > b).astype(float) + .5 * (a == b)
            name = column.removeprefix('score_')
            matrices[name] = matrix
            boot[name] = matrix[ip[:, :, None], jn[:, None, :]].mean(axis=(1, 2))
            top10 = int(group.sort_values(column, ascending=False).head(10).target.sum())
            details.append({**rec, 'model': name, 'auc': float(matrix.mean()), 'auc_low95': float(np.quantile(boot[name], .025)), 'auc_high95': float(np.quantile(boot[name], .975)), 'mounted_at_10': top10})
        delta = boot['global_dinov3'] - boot['region_families']
        rec.update({'dino_minus_region_auc': float(matrices['global_dinov3'].mean()-matrices['region_families'].mean()), 'delta_low95': float(np.quantile(delta,.025)), 'delta_high95': float(np.quantile(delta,.975))})
        results.append(rec)
    pd.DataFrame(details).to_csv(out/'model_metrics.csv',index=False)
    df[['canonical_id','target','community_leiden_fused']].to_csv(out/'cohort_membership.csv',index=False)
    summary = {'analysis':'post-hoc frozen-score within-community control', 'bootstrap_draws':10000, 'seed':20260928, 'results': results, 'input_sha256': {str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pred,membership]}, 'scope':'Single-agent transductive cohort. Controls coarse global community only; does not establish causal context dependence or independent source generalization.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    run()
