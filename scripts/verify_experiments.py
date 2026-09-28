"""Recalculate experimental endpoints from saved predictions and inspection selections."""
from pathlib import Path
import csv
import json
import math
import sys

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))

def main():
    e = P / 'experiments/final_study_20260928/exposure_state_holdout_r1'
    rows = list(csv.DictReader((e / 'blind_test_predictions.csv').open(encoding='utf-8')))
    summary = json.loads((e / 'evaluation_summary.json').read_text())
    assert len(rows) == 48 and len({r['canonical_id'] for r in rows}) == 48
    assert sum(int(r['target']) for r in rows) == 10
    recalculated = {}
    for name, reported in summary['models'].items():
        column = 'score_' + name
        positive = [float(r[column]) for r in rows if int(r['target']) == 1]
        negative = [float(r[column]) for r in rows if int(r['target']) == 0]
        auc = sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))
        ranked = sorted(rows, key=lambda r: -float(r[column]))
        found = sum(int(r['target']) for r in ranked[:10])
        assert math.isclose(auc, reported['roc_auc'], abs_tol=1e-12)
        assert found == reported['mounted_found_at_10']
        recalculated[name] = {'auc': auc, 'found_at_10': found}
    control_dir = P / 'experiments/final_study_20260928/exposure_state_community_control_r1'
    memberships = {r['canonical_id']: int(r['community_leiden_fused']) for r in csv.DictReader((control_dir / 'cohort_membership.csv').open())}
    reported_controls = list(csv.DictReader((control_dir / 'model_metrics.csv').open()))
    for reported in reported_controls:
        selected = [r for r in rows if memberships[r['canonical_id']] == int(reported['community'])]
        column = 'score_' + reported['model']
        positive = [float(r[column]) for r in selected if int(r['target']) == 1]
        negative = [float(r[column]) for r in selected if int(r['target']) == 0]
        auc = sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))
        assert math.isclose(auc, float(reported['auc']), abs_tol=1e-12)
    for receipt in ['functional_test_receipt.json', 'priority_test_receipt.json']:
        data = json.loads((P / 'experiments/final_study_20260928/app_priority_test' / receipt).read_text())
        assert data['status'] == 'pass'
    from solution.evidence_intake.planner import plan_inspection, load_policy
    catalog = json.loads((P/'solution/evidence_intake/demo/collection_catalog.json').read_text(encoding='utf-8'))
    weights = load_policy(P/'experiments/final_study_20260928/region_family_census_r2/family_census.csv')
    plan = plan_inspection(catalog, 50, weights=weights)
    assert plan['after'] == {'families_seen':77,'families_repeated':54}
    assert len({x['canonical_id'] for x in plan['items']}) == 50
    reported_plan = json.loads((P/'experiments/final_study_20260928/inspection_planning_r1/plan.json').read_text(encoding='utf-8'))
    assert [x['canonical_id'] for x in plan['items']] == [x['canonical_id'] for x in reported_plan['items'][:50]]
    from collections import defaultdict, Counter
    followup = P/'experiments/final_study_20260928/inspection_context_coverage_r1'
    selections = list(csv.DictReader((followup.parent/'inspection_planning_r1/selected_photos.csv').open()))
    by_photo = defaultdict(list)
    for region in catalog['items']:
        by_photo[region['parent_id']].append(region)
    bridges = {}
    for method in ['discovery_greedy', 'global_centroids']:
        contexts = defaultdict(set)
        for row in selections:
            if row['method'] == method and int(row['rank']) <= 50:
                for region in by_photo[row['canonical_id']]:
                    contexts[region['family_internal']].add(region['global_community'])
        bridges[method] = {family for family, values in contexts.items() if len(values) >= 2}
    assert len(bridges['discovery_greedy']) == 25 and len(bridges['global_centroids']) == 5
    adjudication = list(csv.DictReader((followup/'pair_adjudication.csv').open(encoding='utf-8')))
    assert len(adjudication) == 25 and {int(r['family_internal']) for r in adjudication} == bridges['discovery_greedy']
    assert Counter(r['decision'] for r in adjudication) == {'supported':10, 'ambiguous':5, 'rejected':10}
    pairs = json.loads((followup/'pair_review_manifest.json').read_text(encoding='utf-8'))
    assert len(pairs) == 50
    assert Counter(r['family_internal'] for r in pairs) == {family:2 for family in bridges['discovery_greedy']}
    pair_dir = followup.parent/'context_pair_selection_r1'
    scored = list(csv.DictReader((pair_dir/'eligible_pair_scores.csv').open()))
    chosen = list(csv.DictReader((pair_dir/'selected_pairs.csv').open()))
    review = json.loads((pair_dir/'review_manifest.json').read_text(encoding='utf-8'))
    judged = {r['review_id']: r['decision'] for r in csv.DictReader((pair_dir/'visual_adjudication.csv').open(encoding='utf-8'))}
    assert len(scored) == 81 and len(review) == len(judged) == 37
    assert {r['review_id'] for r in review} == set(judged)
    by_pair = {r['pair_id']: judged[r['review_id']] for r in review}
    all_regions = {r['region_id']:r for r in catalog['items']}
    batch = {x['canonical_id'] for x in plan['items']}
    for row in scored:
        a,b=all_regions[row['left_id']],all_regions[row['right_id']]
        assert a['parent_id'] in batch and b['parent_id'] in batch
        assert a['parent_id'] != b['parent_id'] and a['source_sha256'] != b['source_sha256']
        assert a['global_community'] != b['global_community']
        assert a['family_internal'] == b['family_internal'] == int(row['family_internal'])
    for family in bridges['discovery_greedy']:
        pool=[r for r in scored if int(r['family_internal'])==family]
        for row in pool:
            ranks=[(sum(float(s[k]) < float(row[k]) for s in pool) +
                    (sum(float(s[k]) == float(row[k]) for s in pool)+1)/2)/len(pool)
                   for k in ['foreground','box','siglip']]
            # Original rank arrays retain float32 precision in this runtime.
            assert math.isclose(min(ranks),float(row['min_rank']),abs_tol=1e-7)
            assert math.isclose(sum(ranks)/3,float(row['mean_rank']),abs_tol=1e-7)
        for method,fields in [('foreground_nearest',['foreground']),('view_consensus',['min_rank','mean_rank','foreground'])]:
            best=min(pool,key=lambda r:tuple(-float(r[k]) for k in fields)+(r['left_id'],r['right_id']))
            actual=next(r for r in chosen if r['method']==method and int(r['family_internal'])==family)
            assert (actual['left_id'],actual['right_id'])==(best['left_id'],best['right_id'])
    for method in ['first_encounter','foreground_nearest','view_consensus']:
        rows_method=[r for r in chosen if r['method']==method]
        assert len(rows_method)==25 and {int(r['family_internal']) for r in rows_method}==bridges['discovery_greedy']
        counts=Counter(by_pair[r['pair_id']] for r in rows_method)
        expected={'supported':10,'ambiguous':5,'rejected':10} if method=='first_encounter' else {'supported':10,'ambiguous':6,'rejected':9}
        assert counts==expected
    style_dir = followup.parent/'exposure_style_control_r1'
    style = list(csv.DictReader((style_dir/'joined_analysis.csv').open()))
    style_summary = json.loads((style_dir/'summary.json').read_text())
    assert len(style) == 48 and {r['canonical_id'] for r in style} == {r['canonical_id'] for r in rows}
    original = {r['canonical_id']:r for r in rows}
    reps = {}
    for row in sorted(style, key=lambda r:r['source_sha256']):
        assert row['target'] == original[row['canonical_id']]['target']
        for model in summary['models']:
            assert math.isclose(float(row['score_'+model]),float(original[row['canonical_id']]['score_'+model]),rel_tol=0,abs_tol=1e-15)
        reps.setdefault(row['scene_group'],row)
    assert len(reps) == 40
    checked = [list(reps.values()), [r for r in reps.values() if r['presentation']=='scene']]
    for selected, key, expected_n in zip(checked,['scene_deduplicated_all','scene_deduplicated_presentation_scene'],[40,30]):
        assert len(selected) == expected_n
        for model, result in style_summary['results'][key]['models'].items():
            pos=[float(r['score_'+model]) for r in selected if int(r['target'])==1]
            neg=[float(r['score_'+model]) for r in selected if int(r['target'])==0]
            auc=sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
            assert math.isclose(auc,result['auc'],abs_tol=1e-12)
    print(json.dumps({'status':'verified','scope':'saved-output recomputation and deterministic planning; not independent validation','models':recalculated,'inspection':plan['after'],'style_control':{'frozen_photos':48,'scene_representatives':40,'scene_presentation_representatives':30}},indent=2))

if __name__ == '__main__':
    main()
