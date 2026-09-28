"""Paired scripted-feedback test on the frozen full inspection catalog."""
from pathlib import Path
import json,sys,tempfile
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from solution.evidence_intake.inspection import InspectionSessions
from solution.evidence_intake.planner import load_policy

def main():
    e=P/'experiments/final_study_20260928'
    catalog=json.loads((P/'solution/evidence_intake/demo/collection_catalog.json').read_text())
    weights=load_policy(e/'region_family_census_r2/family_census.csv')
    out={}
    with tempfile.TemporaryDirectory() as temp:
        for name in ['all_recorded','second_photo_excluded']:
            store=InspectionSessions(Path(temp)/(name+'.sqlite3'),catalog,weights)
            s=store.create(5);first=[r['canonical_id'] for r in s['plan']['items']]
            for i,p in enumerate(first):
                outcome='excluded' if name=='second_photo_excluded' and i==1 else 'recorded'
                s=store.review(s['session_id'],s['version'],p,outcome,'Scripted scenario, not an expert label')
            s=store.next_batch(s['session_id'],s['version']);second=[r['canonical_id'] for r in s['plan']['items']]
            assert not set(first)&set(second)
            out[name]={'initial_photo_ids':first,'next_photo_ids':second,'before':s['plan']['before'],'projected_after_next_batch':s['plan']['after'],'review_outcomes':{p:r['outcome'] for p,r in s['reviews'].items()}}
    assert out['all_recorded']['initial_photo_ids']==out['second_photo_excluded']['initial_photo_ids']
    out['changed_next_batch_positions']=sum(a!=b for a,b in zip(out['all_recorded']['next_photo_ids'],out['second_photo_excluded']['next_photo_ids']))
    out['different_next_batch_photos']=len(set(out['all_recorded']['next_photo_ids'])-set(out['second_photo_excluded']['next_photo_ids']))
    out['scope']='Paired scripted application behavior; catalog memberships are not expert-confirmed component labels. Projected coverage assumes subsequent completion.'
    d=e/'inspection_worklist_r1';d.mkdir(exist_ok=True)
    (d/'feedback_counterfactual.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
