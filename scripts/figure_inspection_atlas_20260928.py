"""Render source-linked scientific figures from frozen graph and evaluation tables."""
from pathlib import Path
import argparse, gzip, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import ConnectionPatch
from matplotlib import patheffects as pe
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from evidence_image_rendering import load_photo, decode_mask, crop_photo_and_mask, draw_whole_evidence, style_image_axis

P=Path(__file__).resolve().parents[1]
E=P/'experiments/final_study_20260928'
G=P/'experiments/multiscale_graph_20260928/global'
COLORS=['#587680','#A3463C','#547BA2','#9171A9','#499781','#B49445','#7F8772','#D17A4F','#286F67','#AC685D','#7495A2','#AE815A','#A19537','#836437','#626F94']
NAMES=['Rakitan dan\ntumpukan','Ponsel','Tetikus','Layar dan\ntelevisi','Mesin cuci','Pencetak','Radio dan\npemutar media','Microwave','Papan\nrangkaian','Papan ketik','Monitor di meja','Laptop','Baterai portabel','Aki kendaraan','Stopkontak\ndan sakelar']
LABELS=[(1.4,-.2),(3.7,3.9),(3.0,-4.45),(.4,6.85),(-4.3,7.5),(-3.5,3.7),(-4.25,.0),(-7.8,5.05),(.1,-7.3),(-.6,2.25),(7.0,.6),(7.0,7.0),(-6.75,-5.6),(-5.0,-9.65),(6.9,-5.2)]

def save(fig,out,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'{name}.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    im=Image.open(out/f'{name}.png')
    im=im.resize((886,round(im.height*886/im.width)),Image.Resampling.LANCZOS)
    im.save(out/f'{name}_print.png'); im.convert('L').save(out/f'{name}_gray.png')

def graph(ax,d,e,labels=True):
    xy=d[['layout_x','layout_y']].to_numpy(); c=d.community_leiden_fused.to_numpy()
    u=e.source_row.to_numpy(int);v=e.target_row.to_numpy(int);inside=c[u]==c[v]
    ax.add_collection(LineCollection(xy[np.stack([u[inside],v[inside]],axis=1)],colors='#8C969B',linewidths=.18,alpha=.23,zorder=0))
    cross=(~inside)&e.kind.eq('maximum_spanning_forest').to_numpy()
    ax.add_collection(LineCollection(xy[np.stack([u[cross],v[cross]],axis=1)],colors='#ABB3B6',linewidths=.35,alpha=.5,zorder=0))
    for i in range(15):
        take=c==i; ax.scatter(xy[take,0],xy[take,1],s=2.8,c=COLORS[i],alpha=.8,edgecolors='white',linewidths=.1,zorder=2)
        if labels:
            a=ax.annotate(NAMES[i],xy=np.median(xy[take],axis=0),xytext=LABELS[i],ha='center',va='center',fontsize=6.5,color=COLORS[i],fontweight='bold',arrowprops={'arrowstyle':'-','lw':.5,'color':COLORS[i]},zorder=5)
            a.set_path_effects([pe.withStroke(linewidth=2,foreground='white')])
    ax.set(xlim=(-9.5,9.2),ylim=(-10.4,8.1),aspect='equal');ax.axis('off')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image-root',type=Path,required=True);ap.add_argument('--proposals',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    d=pd.read_csv(G/'canonical_assignments.csv');e=pd.read_csv(G/'display_edges.csv')
    audit=pd.read_csv(E/'multiview_region_reranker_r1/pair_audit_adjudicated.csv').set_index('audit_index')
    rows=[audit.loc[i] for i in [1,4,13]]
    # The condition example is an explicitly adjudicated photo pair, independent
    # of the retrieval audit's original query/candidate roles.
    condition_path=E/'context_pair_selection_r1/figure_condition_pair.json'
    condition=json.loads(condition_path.read_text(encoding='utf-8'))
    replacement=rows[1].copy()
    for side in ['query','candidate']:
        for key,value in condition[side].items():replacement[f'{side}_{key}']=value
    rows[1]=replacement
    ids={r[f'{s}_region_id'] for r in rows for s in ['query','candidate']};props={}
    with gzip.open(a.proposals,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['region_id'] in ids:props[r['region_id']]=r
    def evidence(row,side,ax,color,crop=False):
        p=props[row[f'{side}_region_id']]
        im=load_photo(a.image_root/row[f'{side}_source_relative_path'],row[f'{side}_source_sha256'])
        if crop:
            m=decode_mask(p)
            if m.shape!=(im.height,im.width):m=np.asarray(Image.fromarray(m*255).resize(im.size,Image.Resampling.NEAREST))>127
            im,m,_=crop_photo_and_mask(im,m,p['mask_bbox_xyxy_native'],.2)
            ax.imshow(im);ax.contour(m,levels=[.5],colors=[color],linewidths=.65);style_image_axis(ax,color)
        else:draw_whole_evidence(ax,im,p,color)
    # Method: original images and native graph, with only operation names.
    fig=plt.figure(figsize=(5.9055,3.25))
    ax=fig.add_axes([.02,.32,.23,.57]);evidence(rows[2],'candidate',ax,'#536B76')
    ax=fig.add_axes([.34,.61,.19,.29]);evidence(rows[2],'candidate',ax,'#C17A22',True)
    ax=fig.add_axes([.31,.10,.26,.36]);graph(ax,d,e,False)
    for start,end in [((.26,.63),(.33,.75)),((.26,.49),(.31,.31)),((.55,.75),(.69,.75)),((.58,.30),(.69,.30))]:
        fig.add_artist(ConnectionPatch(start,end,'figure fraction','figure fraction',arrowstyle='->',lw=.8,color='#40474A'))
    fig.text(.025,.95,'Citra asli',fontsize=8,weight='bold')
    fig.text(.32,.95,'Representasi visual',fontsize=8,weight='bold')
    fig.text(.71,.95,'Penggunaan',fontsize=8,weight='bold')
    fig.text(.34,.56,'Region / mask',fontsize=7)
    fig.text(.32,.04,'Graf citra utuh',fontsize=7)
    fig.text(.71,.73,'Cakupan pola\nantarbagian',fontsize=8,va='center',linespacing=1.6)
    fig.text(.71,.27,'Prioritas relasi\nperakitan',fontsize=8,va='center',linespacing=1.6)
    save(fig,o,'method')
    # Full canonical graph plus original photographs tied to exact graph nodes.
    fig=plt.figure(figsize=(5.9055,6.9));ax=fig.add_axes([.02,.25,.96,.73]);graph(ax,d,e)
    titles=['Kemasan sirkuit','Retak layar','Papan rangkaian'];colors=['#147D78','#B44A5A','#C17A22']
    for n,(row,title,color) in enumerate(zip(rows,titles,colors)):
        x=.025+n*.325
        fig.text(x,.218,chr(97+n)+'  '+title,fontsize=7.3,weight='bold',color=color)
        for j,side in enumerate(['query','candidate']):
            ia=fig.add_axes([x+j*.154,.025,.145,.17]);evidence(row,side,ia,color)
            r=d[d.source_sha256==row[f'{side}_source_sha256']].iloc[0]
            ax.scatter([r.layout_x],[r.layout_y],s=25,facecolors='none',edgecolors=color,lw=1,zorder=7)
            # Dotted leaders are references to photographs, never graph edges.
            con=ConnectionPatch((r.layout_x,r.layout_y),(.5,1),'data','axes fraction',axesA=ax,axesB=ia,color=color,lw=.45,ls=':',alpha=.65,zorder=1)
            fig.add_artist(con)
    save(fig,o,'collection')
    # Exact local evidence with a quantitative mechanism test.
    fig=plt.figure(figsize=(5.9055,6.9))
    for n,(row,title,color) in enumerate(zip(rows,['Kemasan sirkuit terpadu','Retak permukaan layar','Papan pada rakitan dan tumpukan'],colors)):
        y=.80-n*.215
        fig.text(.025,y+.165,chr(97+n)+'  '+title,fontsize=8,weight='bold')
        for x,side in [(.025,'query'),(.54,'candidate')]:
            ax=fig.add_axes([x,y,.265,.15]);evidence(row,side,ax,color)
            ca=fig.add_axes([x+.277,y,.145,.15]);evidence(row,side,ca,color,True)
        fig.add_artist(plt.Line2D([.466,.524],[y+.079]*2,transform=fig.transFigure,color=color,lw=.8))
    cohort=pd.read_csv(E/'exposure_state_community_control_r1/cohort_membership.csv')
    pred=pd.read_csv(E/'exposure_state_holdout_r1/blind_test_predictions.csv')
    pred=pred.merge(cohort[['canonical_id','community_leiden_fused']],on='canonical_id',validate='one_to_one')
    pred=pred[pred.community_leiden_fused.eq(0)].copy()
    assert len(pred)==43 and pred.target.sum()==10
    ax=fig.add_axes([.105,.075,.355,.225]);ay=fig.add_axes([.615,.075,.355,.225])
    ax.set_title('d  Diskriminasi perakitan',loc='left',fontsize=8,pad=9,weight='bold')
    ay.set_title('e  Hasil tinjauan berurutan',loc='left',fontsize=8,pad=9,weight='bold')
    for name,label,color,ls in [('global_dinov3','Citra utuh','#365F91','-'),('region_families','Keluarga region','#C17A22','--')]:
        scores=pred['score_'+name]
        fpr,tpr,_=roc_curve(pred.target,scores)
        auc=roc_auc_score(pred.target,scores)
        ax.step(fpr,tpr,where='post',lw=1.5,color=color,ls=ls,label=label+' ('+f'{auc:.3f}'.replace('.',',')+')')
        ordered=pred.sort_values('score_'+name,ascending=False,kind='stable')
        ay.step(np.arange(44),np.r_[0,ordered.target.cumsum()],where='post',lw=1.5,color=color,ls=ls)
    ax.plot([0,1],[0,1],color='#AAAAAA',ls=':',lw=.7,zorder=0)
    ax.set(xlim=(-.025,1.025),ylim=(-.025,1.055),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='False positive rate',ylabel='True positive rate')
    ax.legend(loc='lower right',frameon=False,fontsize=6.2,handlelength=2)
    ay.set(xlim=(0,43),ylim=(0,10.5),xticks=[0,10,20,30,43],yticks=[0,5,10],xlabel='Foto yang ditinjau',ylabel='Kasus terpasang ditemukan')
    ax.tick_params(labelsize=7);ay.tick_params(labelsize=7)
    save(fig,o,'relations')
    # Comparative coverage, with the conservative-evidence result equally visible.
    t=pd.read_csv(E/'inspection_planning_r1/evaluation.csv')
    fig,axs=plt.subplots(1,2,figsize=(5.9055,3.3));fig.subplots_adjust(left=.10,right=.985,bottom=.29,top=.87,wspace=.33)
    styles=[('discovery_greedy','Discovery','#147D78','-','o'),('global_centroids','Wakil global','#365F91','--','s'),('static_local_degree','Bobot statis','#B46C38','-.','^'),('random','Acak','#777777',':','D')]
    for ax,col,title in zip(axs,['families_seen','r3_concepts_seen'],['a  Keluasan eksplorasi','b  Dukungan konservatif']):
        for method,label,color,ls,marker in styles:
            sub=t[t.method==method];g=sub.groupby('budget')[col]
            mean=g.mean();ax.plot(mean.index,mean.values,label=label,color=color,ls=ls,marker=marker,ms=3,lw=1.2)
            if method=='random':ax.fill_between(mean.index,g.quantile(.025),g.quantile(.975),color=color,alpha=.13)
        ax.set_title(title,loc='left',fontsize=8,pad=10,fontweight='bold');ax.set_xlabel('Jumlah foto yang dipilih');ax.set_xticks([10,25,50,100]);ax.tick_params(labelsize=7)
        ax.set_ylim(0,110 if col=='families_seen' else 17)
        ax.set_ylabel('Keluarga region' if col=='families_seen' else 'Konsep didukung')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.035),ncol=4,frameon=False,fontsize=7,columnspacing=1)
    save(fig,o,'inspection')
    sources=[G/'canonical_assignments.csv',G/'display_edges.csv',E/'multiview_region_reranker_r1/pair_audit_adjudicated.csv',E/'inspection_planning_r1/evaluation.csv',E/'exposure_state_community_control_r1/summary.json',E/'exposure_state_community_control_r1/model_metrics.csv']
    sources += [E/'exposure_state_community_control_r1/cohort_membership.csv',E/'exposure_state_holdout_r1/blind_test_predictions.csv']
    sources += [condition_path,E/'context_pair_selection_r1/visual_adjudication.csv']
    sources=[s for s in sources if s.exists()]
    (o/'figure_sources.json').write_text(json.dumps({'inputs':[{ 'path':str(s.relative_to(P)).replace('\\','/'),'sha256':hashlib.sha256(s.read_bytes()).hexdigest()} for s in sources], 'photo_region_ids':sorted(ids),'layout':'unchanged frozen coordinates; dotted lines are photo references'},indent=2),encoding='utf-8')
    print('Four source-linked scientific figures rendered; visual review pending.')
if __name__=='__main__':main()
