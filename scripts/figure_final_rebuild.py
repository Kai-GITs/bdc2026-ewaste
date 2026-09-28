"""Publication figures from frozen discovery, source photos, masks and predictions."""
from pathlib import Path
import argparse,json,gzip,hashlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image
from sklearn.metrics import roc_curve,roc_auc_score
from evidence_image_rendering import load_photo,decode_mask,crop_photo_and_mask,draw_whole_evidence,style_image_axis
from figure_inspection_atlas_20260928 import graph,save,COLORS,NAMES
P=Path(__file__).resolve().parents[1];E=P/'experiments/final_study_20260928';G=P/'experiments/multiscale_graph_20260928/global'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--image-root',type=Path,required=True);ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 selection=E/'presentation_rebuild_final/photo_selection.json';s=json.loads(selection.read_text());d=pd.read_csv(G/'canonical_assignments.csv');edges=pd.read_csv(G/'display_edges.csv')
 audit=pd.read_csv(E/'multiview_region_reranker_r1/pair_audit_adjudicated.csv').set_index('audit_index');ic=audit.loc[1];cond=json.loads((E/'context_pair_selection_r1/figure_condition_pair.json').read_text());pairs=[[{k:ic[f'{side}_{k}'] for k in ['region_id','source_relative_path','source_sha256']} for side in ['query','candidate']],[cond[side] for side in ['query','candidate']]]
 board=audit.loc[13];graph_pairs=pairs+[[{k:board[f'{side}_{k}'] for k in ['region_id','source_relative_path','source_sha256']} for side in ['query','candidate']]]
 wanted={r['region_id'] for pair in graph_pairs for r in pair}|{s['method']['region_id']}|{rid for r in s['inspection_trace'] for rid in r['selected_region_ids']};props={}
 with gzip.open(a.proposals,'rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line)
   if r['region_id'] in wanted:props[r['region_id']]=r
 assert wanted==set(props)
 used=[]
 def photo(r):
  path=a.image_root/r['source_relative_path']
  if len(str(path))>=260:path=Path('\\\\?\\'+str(path.resolve()))
  im=load_photo(path,r['source_sha256']);used.append({k:r[k] for k in ['source_relative_path','source_sha256']});return im
 def evidence(ax,r,color='#147d78',crop=False):
  im=photo(r);p=props[r['region_id']]
  if not crop:draw_whole_evidence(ax,im,p,color);return
  m=decode_mask(p)
  if m.shape!=(im.height,im.width):m=np.asarray(Image.fromarray(m*255).resize(im.size,Image.Resampling.NEAREST))>127
  im,m,_=crop_photo_and_mask(im,m,p['mask_bbox_xyxy_native'],.2);ax.imshow(im);ax.contour(m,levels=[.5],colors=[color],linewidths=.7);style_image_axis(ax,color)
 # One new photo, two actual measurement scales.
 fig=plt.figure(figsize=(5.9055,2.8));r=s['method']
 for x,title,crop in [(.02,'Citra tumpukan',False),(.365,'Region papan',False),(.71,'Detail papan',True)]:
  ax=fig.add_axes([x,.19,.27,.65]);fig.text(x,.92,title,fontsize=9,weight='bold')
  if x==.02:ax.imshow(photo(r));ax.axis('off')
  else:evidence(ax,r,crop=crop)
 for x in [.31,.65]:fig.add_artist(ConnectionPatch((x,.53),(x+.035,.53),'figure fraction','figure fraction',arrowstyle='->',lw=.8,color='#555'))
 fig.text(.02,.055,'Citra utuh',fontsize=8);fig.text(.365,.055,'Mask segmentasi',fontsize=8);fig.text(.71,.055,'Area terpilih',fontsize=8);save(fig,a.output,'method')
 # Full collection linked to the actual local evidence preferred in the review.
 NAMES[13]='Aki dan\nbaterai blok'
 assignments_path=E/'region_family_graph_object_filtered_r2/assignments.csv.gz'
 region_edges_path=E/'region_family_graph_object_filtered_r2/edges.csv.gz'
 assignments=pd.read_csv(assignments_path);region_edges=pd.read_csv(region_edges_path)
 region_lookup=assignments.set_index('region_id');parent_lookup=d.set_index('canonical_id')
 local_graph_records=[]
 fig=plt.figure(figsize=(5.9055,7.5));ax=fig.add_axes([.02,.36,.96,.63]);graph(ax,d,edges)
 for n,(pair,title,color) in enumerate(zip(graph_pairs,['Kemasan sirkuit','Retak layar','Papan rangkaian'],['#147D78','#B44A5A','#C17A22'])):
  family=int(region_lookup.loc[pair[0]['region_id'],'main_family_internal'])
  assert all(int(region_lookup.loc[r['region_id'],'main_family_internal'])==family for r in pair)
  members=assignments[assignments.main_family_internal.eq(family)].copy()
  ids=set(members.graph_row.astype(int));inside=region_edges[region_edges.source_graph_row.isin(ids)&region_edges.target_graph_row.isin(ids)]
  local=nx.Graph();local.add_nodes_from(sorted(ids))
  for edge in inside.itertuples():local.add_edge(int(edge.source_graph_row),int(edge.target_graph_row),weight=float(edge.two_view_parent_normalized_weight))
  pos=nx.spring_layout(local,seed=23,weight='weight',iterations=150)
  x=.025+n*.325;fig.text(x,.34,chr(97+n)+'  '+title,fontsize=7.3,weight='bold',color=color)
  la=fig.add_axes([x,.181,.302,.145]);la.set_aspect('equal');la.axis('off')
  nx.draw_networkx_edges(local,pos,ax=la,width=.35,edge_color='#9ca3a7',alpha=.6)
  local_colors=[COLORS[int(parent_lookup.loc[members.set_index('graph_row').loc[g,'parent_id'],'community_leiden_fused'])] for g in local.nodes]
  nx.draw_networkx_nodes(local,pos,ax=la,node_size=8,node_color=local_colors,edgecolors='white',linewidths=.25)
  parents=parent_lookup.loc[members.parent_id.unique()]
  span=parents.community_leiden_fused.nunique()
  fig.text(x+.151,.157,f'{len(members)} region · {len(parents)} foto · {span} komunitas',fontsize=6.8,ha='center',bbox={'facecolor':'white','edgecolor':'none','pad':1},zorder=8)
  if n==1:
   ax.scatter(parents.layout_x,parents.layout_y,s=10,facecolor='none',edgecolor=color,lw=.6,alpha=.8,zorder=6)
  local_graph_records.append({'family_internal':family,'regions':len(members),'photos':len(parents),'global_communities':int(span),'nodes':[{'graph_row':int(g),'region_id':str(members.set_index('graph_row').loc[g,'region_id']),'x':float(pos[g][0]),'y':float(pos[g][1])} for g in local.nodes],'edges':[{'source':int(u),'target':int(v),'weight':float(data['weight'])} for u,v,data in local.edges(data=True)]})
  for j,r in enumerate(pair):
   ia=fig.add_axes([x+j*.154,.015,.145,.125]);evidence(ia,r,color)
   node=d[d.source_sha256==r['source_sha256']].iloc[0]
   ax.scatter(node.layout_x,node.layout_y,s=25,facecolor='none',edgecolor=color,lw=1,zorder=7)
   g=int(region_lookup.loc[r['region_id'],'graph_row']);la.scatter(*pos[g],s=27,facecolor='none',edgecolor=color,lw=.8,zorder=6)
   fig.add_artist(ConnectionPatch((node.layout_x,node.layout_y),pos[g],'data','data',axesA=ax,axesB=la,color=color,lw=.4,ls=':',alpha=.5,zorder=1))
   fig.add_artist(ConnectionPatch(pos[g],(.5,1),'data','axes fraction',axesA=la,axesB=ia,color=color,lw=.55,ls=':',alpha=.85,zorder=1))
 save(fig,a.output,'collection')
 local_graph_path=E/'presentation_rebuild_final/local_family_graph_evidence.json'
 local_graph_path.write_text(json.dumps({'layout':'spring layout of induced frozen family subgraphs; no clustering recomputation','families':local_graph_records},indent=2))
 # Local claims, dedicated photo+native-region evidence only.
 fig=plt.figure(figsize=(5.9055,4.3))
 for i,(pair,title,color) in enumerate(zip(pairs,['a  Kemasan sirkuit pada susunan papan berbeda','b  Retak permukaan pada perangkat berbeda'],['#147d78','#ad4861'])):
  y=.55-i*.48;fig.text(.02,y+.39,title,fontsize=9,weight='bold')
  for j,r in enumerate(pair):
   x=.02+j*.51;evidence(fig.add_axes([x,y,.27,.33]),r,color);evidence(fig.add_axes([x+.282,y,.165,.33]),r,color,True)
  fig.add_artist(plt.Line2D([.477,.516],[y+.16]*2,transform=fig.transFigure,color=color,lw=.8))
 save(fig,a.output,'relations')
 # Original test photographs and 40-scene ROC (frozen scores).
 fig=plt.figure(figsize=(5.9055,2.55))
 for i,(r,title) in enumerate(zip(s['assembly'],['a  Papan di dalam sasis','b  Papan lepas'])):
  x=.02+.51*i;ax=fig.add_axes([x,.05,.46,.80]);ax.imshow(photo(r));ax.axis('off');fig.text(x,.96,title,fontsize=9,weight='bold')
 save(fig,a.output,'assembly_examples')
 fig=plt.figure(figsize=(5.9055,3.1))
 pred=pd.read_csv(E/'exposure_style_control_r1/scene_representatives.csv');assert len(pred)==40
 ax=fig.add_axes([.12,.20,.84,.75])
 for m,label,color,ls in [('global_dinov3','Citra utuh','#365f91','-'),('region_families','Ringkasan keluarga','#bd7e29','--')]:
  fpr,tpr,_=roc_curve(pred.target,pred['score_'+m]);auc=roc_auc_score(pred.target,pred['score_'+m]);ax.step(fpr,tpr,where='post',color=color,lw=1.7,ls=ls,label=f'{label}  {auc:.3f}'.replace('.',','))
 ax.plot([0,1],[0,1],color='#aaa',lw=.8,ls=':');ax.set(xlim=(0,1.01),ylim=(0,1.04),xticks=[0,.25,.5,.75,1],yticks=[0,.5,1],xlabel='False positive rate',ylabel='True positive rate');ax.tick_params(labelsize=10);ax.xaxis.label.set_size(11);ax.yaxis.label.set_size(11);ax.legend(frameon=False,loc='lower right',fontsize=10);save(fig,a.output,'assembly')
 # Inspection utility is encoded quantitatively; unclear patch examples omitted.
 fig=plt.figure(figsize=(5.9055,3.15))
 t=pd.read_csv(E/'inspection_planning_r1/evaluation.csv');axs=[fig.add_axes([.10,.25,.36,.65]),fig.add_axes([.62,.25,.36,.65])]
 styles=[('discovery_greedy','Adaptif','#147D78','-','o'),('global_centroids','Wakil global','#365F91','--','s'),('static_local_degree','Bobot statis','#B46C38','-.','^'),('random','Acak','#777777',':','D')]
 for ax,col in zip(axs,['families_seen','r3_concepts_seen']):
  for method,label,color,ls,marker in styles:
   g=t[t.method==method].groupby('budget')[col];mean=g.mean();ax.plot(mean.index,mean.values,label=label,color=color,ls=ls,marker=marker,ms=3,lw=1.1)
   if method=='random':ax.fill_between(mean.index,g.quantile(.025),g.quantile(.975),color=color,alpha=.13)
  ax.set(xlabel='Anggaran foto',xticks=[10,50,100],ylim=(0,110 if col=='families_seen' else 17),ylabel='Keluarga region' if col=='families_seen' else 'Konsep didukung');ax.tick_params(labelsize=9);ax.xaxis.label.set_size(10);ax.yaxis.label.set_size(10)
 fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.0),ncol=4,frameon=False,fontsize=8.5);save(fig,a.output,'inspection')
 inputs=[selection,G/'canonical_assignments.csv',G/'display_edges.csv',E/'multiview_region_reranker_r1/pair_audit_adjudicated.csv',E/'context_pair_selection_r1/figure_condition_pair.json',E/'exposure_style_control_r1/scene_representatives.csv',E/'inspection_planning_r1/evaluation.csv',E/'inspection_planning_r1/plan.json',assignments_path,region_edges_path,local_graph_path]
 provenance={'inputs':[{'path':p.relative_to(P).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],'photos':list({r['source_sha256']:r for r in used}.values()),'masks':[{'region_id':k,'mask_sha256':v['mask_sha256']} for k,v in props.items()],'selection':selection.relative_to(P).as_posix(),'figure_count':6}
 (a.output/'figure_sources.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8');print('Six figures; original photo and mask hashes verified. Visual review pending.')
if __name__=='__main__':main()

