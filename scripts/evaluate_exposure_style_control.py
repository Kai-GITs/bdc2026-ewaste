"""Frozen-score presentation and repeated-scene controls on all 48 test photos."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODELS=['global_dinov3','global_siglip2','global_fused','region_families','region_plus_global']

def paired_auc(frame, draws=10000, seed=260928):
    pos=frame[frame.target==1];neg=frame[frame.target==0]
    meta=dict(n=len(frame),positive=len(pos),negative=len(neg),estimable=bool(len(pos) and len(neg)))
    if not meta['estimable']:return meta
    a=pos[['score_'+m for m in MODELS]].to_numpy()
    b=neg[['score_'+m for m in MODELS]].to_numpy()
    wins=(a[:,None,:]>b[None,:,:]).astype(float)+.5*(a[:,None,:]==b[None,:,:])
    rng=np.random.default_rng(seed)
    pi=rng.integers(len(pos),size=(draws,len(pos)))
    ni=rng.integers(len(neg),size=(draws,len(neg)))
    bootstrap=wins[pi[:,:,None],ni[:,None,:],:].mean(axis=(1,2))
    point=wins.mean(axis=(0,1))
    meta['models']={m:dict(auc=float(point[k]),low95=float(np.quantile(bootstrap[:,k],.025)),high95=float(np.quantile(bootstrap[:,k],.975))) for k,m in enumerate(MODELS)}
    delta=bootstrap[:,0]-bootstrap[:,3]
    meta['dino_minus_region']=dict(delta=float(point[0]-point[3]),low95=float(np.quantile(delta,.025)),high95=float(np.quantile(delta,.975)))
    return meta

def main():
    base=ROOT/'experiments/final_study_20260928';out=base/'exposure_style_control_r1'
    paths={'protocol':ROOT/'experiments/protocols/exposure_style_control_r1.json',
           'predictions':base/'exposure_state_holdout_r1/blind_test_predictions.csv',
           'review_manifest':out/'review_manifest.json','visual_labels':out/'visual_style_adjudication.csv',
           'cross_split_review':out/'cross_split_review.json'}
    pred=pd.read_csv(paths['predictions']);manifest=pd.DataFrame(json.loads(paths['review_manifest'].read_text(encoding='utf-8')))
    labels=pd.read_csv(paths['visual_labels'])
    assert len(labels)==48 and labels.review_id.is_unique and set(labels.review_id)==set(manifest.review_id)
    frame=pred.merge(manifest[['review_id','canonical_id','width','height']],on='canonical_id',validate='one_to_one').merge(labels,on='review_id',validate='one_to_one')
    assert len(frame)==48 and frame.target.sum()==10
    assert set(frame.presentation)<= {'scene','isolated','composite','uncertain'}
    frame.to_csv(out/'joined_analysis.csv',index=False)
    pd.crosstab(frame.presentation,frame.target).to_csv(out/'presentation_class_counts.csv')
    results={'all_photos':paired_auc(frame)}
    for name,group in frame.groupby('presentation'):
        results['presentation_'+name]=paired_auc(group)
    results['scene_min_side_150']=paired_auc(frame[(frame.presentation=='scene')&(frame[['width','height']].min(axis=1)>=150)])
    # Deterministic source-scene sensitivity; do not average scores to improve them.
    consistency=frame.groupby('scene_group').target.nunique()
    conflicts=consistency[consistency>1].index.tolist()
    reps=frame.sort_values('source_sha256').drop_duplicates('scene_group')
    reps.to_csv(out/'scene_representatives.csv',index=False)
    results['scene_deduplicated_all']=paired_auc(reps)
    results['scene_deduplicated_presentation_scene']=paired_auc(reps[reps.presentation=='scene'])
    source_review=json.loads(paths['cross_split_review'].read_text(encoding='utf-8'))
    unresolved={r['test_review_id'] for r in source_review['candidate_pairs'] if r['decision']=='unresolved'}
    suspect_groups=set(frame.loc[frame.review_id.isin(unresolved),'scene_group'])
    results['scene_deduplicated_excluding_unresolved_source']=paired_auc(reps[~reps.scene_group.isin(suspect_groups)])
    results['scene_deduplicated_presentation_scene_excluding_unresolved_source']=paired_auc(reps[(reps.presentation=='scene')&~reps.scene_group.isin(suspect_groups)])
    numerator=np.zeros(len(MODELS));denominator=0
    for name,group in frame.groupby('presentation'):
        r=paired_auc(group,draws=1)
        if r['estimable']:
            pairs=r['positive']*r['negative'];denominator+=pairs
            numerator+=pairs*np.array([r['models'][m]['auc'] for m in MODELS])
    conditional={'within_presentation_positive_negative_pairs':denominator,'auc':dict(zip(MODELS,(numerator/denominator).tolist()))}
    payload=dict(results=results,conditional_auc=conditional,scene_group_target_conflicts=conflicts,
                 inputs={k:{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},
                 caveat='Adaptive, single-agent descriptive controls. Presentation labels and within-test scene grouping do not prove source independence from development photos or causal context use.')
    (out/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,r in results.items():print(name, json.dumps(r))
    print('conditional',conditional,'target_conflicts',conflicts)
if __name__=='__main__':main()
