"""Summarize original-image pair adjudication without dropping any family."""
from pathlib import Path
import json
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def main():
    out = ROOT / 'experiments/final_study_20260928/context_pair_selection_r1'
    manifest = pd.DataFrame(json.loads((out/'review_manifest.json').read_text(encoding='utf-8')))
    labels = pd.read_csv(out/'visual_adjudication.csv')
    assert set(labels.review_id) == set(manifest.review_id) and labels.review_id.is_unique
    assert set(labels.decision) <= {'supported','ambiguous','rejected'}
    mapping = manifest[['review_id','pair_id']].merge(labels, validate='one_to_one')
    selected = pd.read_csv(out/'selected_pairs.csv').merge(mapping, on='pair_id', validate='many_to_one')
    assert selected.groupby('method').size().eq(25).all()
    selected.to_csv(out/'adjudicated_selections.csv',index=False)
    counts = pd.crosstab(selected.method, selected.decision).reindex(columns=['supported','ambiguous','rejected'],fill_value=0)
    counts['denominator'] = 25
    counts.to_csv(out/'method_counts.csv')
    pivot = selected.assign(supported=selected.decision.eq('supported')).pivot(index='family_internal',columns='method',values='supported')
    comparisons=[]
    for method,baseline in [('foreground_nearest','first_encounter'),('view_consensus','first_encounter'),('view_consensus','foreground_nearest')]:
        comparisons.append(dict(method=method,baseline=baseline,
                                wins=int((pivot[method]&~pivot[baseline]).sum()),
                                losses=int((~pivot[method]&pivot[baseline]).sum()),
                                tied_supported=int((pivot[method]&pivot[baseline]).sum()),
                                tied_not_supported=int((~pivot[method]&~pivot[baseline]).sum())))
    old=pd.read_csv(ROOT/'experiments/final_study_20260928/inspection_context_coverage_r1/pair_adjudication.csv')
    cross=selected[selected.method=='first_encounter'].merge(old[['family_internal','decision']],on='family_internal',suffixes=('_current','_previous'))
    changes=cross[cross.decision_current!=cross.decision_previous][['family_internal','decision_current','decision_previous']].to_dict('records')
    result=dict(method_counts=counts.reset_index().to_dict('records'),paired_comparisons=comparisons,
                baseline_review_changes=changes,reviewer='one Codex agent; adaptive and not independent',
                scope='Broad visible correspondence in a fixed batch; not semantic component identity, source independence or novel discovery count',
                adjudication_sha256=hashlib.sha256((out/'visual_adjudication.csv').read_bytes()).hexdigest())
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
