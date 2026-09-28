"""Compare discovery-based inspection scheduling with matched-budget baselines."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from solution.evidence_intake.planner import plan_inspection,photo_index,load_policy


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--feature-index',type=Path,required=True)
    args=parser.parse_args()
    e=P/'experiments/final_study_20260928'
    paths={'catalog':P/'solution/evidence_intake/demo/collection_catalog.json','census':e/'region_family_census_r2/family_census.csv','core':e/'region_family_consensus_r3/photo_concept_support.csv.gz','global':P/'experiments/multiscale_graph_20260928/global/canonical_assignments.csv'}
    catalog=json.loads(paths['catalog'].read_text(encoding='utf-8'))
    photos=photo_index(catalog); ids=sorted(photos); weights=load_policy(paths['census'])
    start=time.perf_counter(); plan=plan_inspection(catalog,100,weights=weights); seconds=time.perf_counter()-start
    ranked=[i['canonical_id'] for i in plan['items']]
    static=sorted(ids,key=lambda p:(-sum(weights[f] for f in photos[p]['families']),p))
    # Rank by cosine distance to each frozen whole-image community centroid.
    index=pd.read_csv(args.feature_index); features=np.load(args.features,mmap_mode='r')
    features=np.array(features,dtype=np.float32,copy=True)
    features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    sha_column='sha256' if 'sha256' in index else 'source_sha256'
    sha_row=dict(zip(index[sha_column],index['row'].astype(int)))
    global_df=pd.read_csv(paths['global']).set_index('canonical_id')
    global_lists=[]
    for cid in sorted({photos[p]['global_community'] for p in ids}):
        members=[p for p in ids if photos[p]['global_community']==cid]
        rows=[sha_row[global_df.loc[p,'source_sha256']] for p in members]
        matrix=features[rows]; centroid=matrix.mean(axis=0); centroid/=max(np.linalg.norm(centroid),1e-12)
        similarities=matrix@centroid
        global_lists.append(sorted(zip(similarities,members),key=lambda v:(-v[0],v[1])))
    whole=[group[depth][1] for depth in range(max(map(len,global_lists))) for group in global_lists if depth<len(group)]
    support=pd.read_csv(paths['core']);support=support[support.configuration_core_support.astype(bool)]
    core={p:set(g.descriptive_family_key) for p,g in support.groupby('parent_id')}
    all_concepts=set(support.descriptive_family_key)
    random_orders=[np.random.default_rng(seed).permutation(ids).tolist() for seed in range(100)]
    orders={'discovery_greedy':ranked,'global_centroids':whole,'static_local_degree':static}
    results=[]
    for budget in [10,25,50,100]:
        effective=min(budget,len(ranked))
        for method,seqs in [(k,[v]) for k,v in orders.items()]+[('random',random_orders)]:
            for replicate,seq in enumerate(seqs):
                picked=seq[:effective]
                fam_counts={f:sum(f in photos[p]['families'] for p in picked) for f in weights}
                core_counts={c:sum(c in core.get(p,set()) for p in picked) for c in all_concepts}
                results.append({'method':method,'replicate':replicate,'budget':budget,'photos_reviewed':len(picked),
                    'families_seen':sum(n>0 for n in fam_counts.values()),'families_repeated':sum(n>=2 for n in fam_counts.values()),
                    'weighted_coverage':sum(weights[f]*min(n,2)/2 for f,n in fam_counts.items()),
                    'r3_concepts_seen':sum(n>0 for n in core_counts.values()),'r3_concepts_repeated':sum(n>=2 for n in core_counts.values()),
                    'global_communities':len({photos[p]['global_community'] for p in picked})})
    out=e/'inspection_planning_r1';out.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out/'evaluation.csv',index=False)
    summary={'eligible_canonical_photos':len(ids),'canonical_collection_total':3931,'families':129,'positive_utility_families':sum(v>0 for v in weights.values()),
        'core_semantic_concepts':len(all_concepts),'planning_seconds':seconds,'planned_photos':len(ranked),'r3_used_by_planner':False,
        'scope':'Transductive scheduling and proxy coverage; r3 is held aside from planning but is derived from correlated evidence. No human time or physical recycling outcome measured.',
        'inputs':{k:{'path':str(p.relative_to(P)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame([{'method':k,'rank':j+1,'canonical_id':p} for k,seq in orders.items() for j,p in enumerate(seq[:100])]).to_csv(out/'selected_photos.csv',index=False)
    print(json.dumps(summary,indent=2)); print(pd.DataFrame(results).groupby(['method','budget'])[['photos_reviewed','families_seen','families_repeated','r3_concepts_seen','r3_concepts_repeated','global_communities']].mean().to_string())


if __name__=='__main__':run()
