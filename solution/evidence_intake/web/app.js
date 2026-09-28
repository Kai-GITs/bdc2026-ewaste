const $ = id => document.getElementById(id);
const state = {query:null,candidates:[],selectedCandidate:null,group:new Map(),dossier:null};
const imageUrl = path => '/images/' + path.split('/').map(encodeURIComponent).join('/');
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const toast = message => {const node=$('toast');node.textContent=message;node.classList.add('show');setTimeout(()=>node.classList.remove('show'),2600)};
async function request(url, options){const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data}

function evidenceFigure(item, compact=false){
  const [x1,y1,x2,y2]=item.bbox_xyxy;
  return `<figure class="evidence-figure ${compact?'compact':''}" data-width="${item.width}" data-height="${item.height}" data-box="${[x1,y1,x2,y2].join(',')}"><img src="${imageUrl(item.source_relative_path)}" alt="${esc(item.family_name)}"><span class="region-box"></span></figure>`;
}
function alignBoxes(root=document){
  root.querySelectorAll('.evidence-figure').forEach(figure=>{
    const image=figure.querySelector('img'),box=figure.querySelector('.region-box');
    const place=()=>{
      const width=Number(figure.dataset.width),height=Number(figure.dataset.height);
      if(!image.complete||!image.naturalWidth||!width||!height)return;
      // Small originals are not enlarged by CSS; use the rendered image bounds.
      const frame=figure.getBoundingClientRect(),drawn=image.getBoundingClientRect();
      const sx=drawn.width/width,sy=drawn.height/height;
      const [x1,y1,x2,y2]=figure.dataset.box.split(',').map(Number);
      Object.assign(box.style,{left:`${drawn.left-frame.left+x1*sx}px`,top:`${drawn.top-frame.top+y1*sy}px`,width:`${(x2-x1)*sx}px`,height:`${(y2-y1)*sy}px`});
    };
    image.addEventListener('load',place,{once:true});place();
  });
}
const supportLabel = item => item.is_r3_core?'dukungan konsensus':'pola hasil pengelompokan';
const contextRelation = (query,candidate) => query.global_community===candidate.global_community?'konteks global serupa':'melintasi konteks global';

function renderExamples(items){
  $('exampleList').innerHTML=items.slice(0,12).map(item=>`<button class="example" data-query="${item.region_id}">${evidenceFigure(item,true)}<span><strong>${esc(item.family_name)}</strong><small>${esc(item.global_context_name)} · ${supportLabel(item)}</small></span></button>`).join('');
  document.querySelectorAll('[data-query]').forEach(button=>button.onclick=()=>selectQuery(button.dataset.query).catch(error=>toast(error.message)));alignBoxes($('exampleList'));
}
async function loadFamily(familyInternal){const page=await request(`/api/collection/items?family_internal=${encodeURIComponent(familyInternal)}&limit=12`);renderExamples(page.items)}
async function selectQuery(regionId){
  const result=await request(`/api/collection/query?region_id=${encodeURIComponent(regionId)}`);state.query=result.query;state.candidates=result.candidates;state.selectedCandidate=result.candidates[0]||null;state.group.clear();state.group.set(state.query.region_id,state.query);$('compareEmpty').classList.add('hidden');$('comparison').classList.remove('hidden');$('exportBlock').classList.add('hidden');renderComparison();renderGroup();
}
function renderComparison(){
  const query=state.query,candidate=state.selectedCandidate;$('queryFigure').innerHTML=evidenceFigure(query);$('queryFacts').innerHTML=`<strong>${esc(query.family_name)}</strong><span>${esc(query.global_context_name)}</span><span>${supportLabel(query)}</span>`;$('candidateFigure').innerHTML=candidate?evidenceFigure(candidate):'<div class="empty">Tidak ada kandidat.</div>';$('candidateFacts').innerHTML=candidate?`<strong>${esc(candidate.family_name)}</strong><span>${esc(candidate.global_context_name)}</span><span>${contextRelation(query,candidate)} · kemiripan ${candidate.similarity.toFixed(3)}</span>`:'';
  $('candidateList').innerHTML=state.candidates.map((item,index)=>`<button class="candidate ${state.selectedCandidate?.region_id===item.region_id?'active':''}" data-candidate="${item.region_id}">${evidenceFigure(item,true)}<span class="rank">${index+1}</span><span class="candidate-copy"><strong>${esc(item.family_name)}</strong><small>${item.same_family?'keluarga sama':'keluarga lain'} · ${contextRelation(query,item)}</small></span></button>`).join('');document.querySelectorAll('[data-candidate]').forEach(button=>button.onclick=()=>{state.selectedCandidate=state.candidates.find(item=>item.region_id===button.dataset.candidate);renderComparison()});alignBoxes($('comparison'));
}
function renderGroup(){
  const rows=[...state.group.values()];$('groupList').innerHTML=rows.length?rows.map(item=>`<div class="group-row">${evidenceFigure(item,true)}<div><strong>${esc(item.family_name)}</strong><small>${esc(item.global_context_name)}</small><small>${supportLabel(item)}</small><details><summary>Provenance</summary><code>${esc(item.region_id)} · ${esc(item.source_sha256.slice(0,12))}…</code></details></div><button class="remove" data-remove="${item.region_id}" aria-label="Hapus">×</button></div>`).join(''):'<p class="empty-inline">Belum ada region dipilih.</p>';document.querySelectorAll('[data-remove]').forEach(button=>button.onclick=()=>{state.group.delete(button.dataset.remove);renderGroup()});$('createGroup').disabled=!rows.length;$('createGroup').textContent=`Buat berkas bukti${rows.length?` (${rows.length} region)`:''}`;alignBoxes($('groupList'));
}
function addSelectedCandidate(){if(!state.selectedCandidate)return;state.group.set(state.selectedCandidate.region_id,state.selectedCandidate);renderGroup();toast('Kandidat ditambahkan ke grup')}
async function matchUpload(file){const buffer=await file.arrayBuffer(),hash=await crypto.subtle.digest('SHA-256',buffer);const sha=[...new Uint8Array(hash)].map(value=>value.toString(16).padStart(2,'0')).join('');const result=await request(`/api/collection/photo?sha256=${sha}`);renderExamples(result.items);$('sourceMessage').textContent=`Foto ditemukan: ${result.items.length} region tersedia untuk diperiksa.`;$('sourceMessage').classList.remove('hidden')}
async function createDossier(){
  const regionIds=[...state.group.keys()],note=$('groupNote').value.trim(),status=$('evidenceStatus').value;let dossier=await request('/api/collection/dossiers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({region_ids:regionIds,batch_id:`BDC-GRUP-${new Date().toISOString().replace(/[:.]/g,'-')}`})});if(status!=='model_predicted_visible'||note){const corrections=dossier.state.entries.map(entry=>({entry_id:entry.entry_id,changes:{evidence_status:status,review_status:status==='observed_manual'?'reviewed':'needs_review',review_note:note}}));dossier=await request(`/api/dossiers/${encodeURIComponent(dossier.dossier_id)}/corrections`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:'candidate group review',corrections})})}state.dossier=dossier;const base=`/api/dossiers/${encodeURIComponent(dossier.dossier_id)}/export?format=`;$('exportJson').href=base+'json';$('exportCsv').href=base+'csv';$('exportPdf').href=base+'pdf';$('exportSummary').textContent=`${dossier.state.batch_id} · revisi ${dossier.revision} · ${dossier.state.entries.length} entri dengan provenance region.`;$('exportBlock').classList.remove('hidden');toast('Berkas bukti tersimpan')
}
async function init(){
  const [summary,families]=await Promise.all([request('/api/collection/summary'),request('/api/collection/families')]);$('collectionMeta').textContent=`${summary.canonical_photos.toLocaleString('id-ID')} foto · ${summary.r2_families} keluarga visual`;$('familySelect').innerHTML=families.map(item=>`<option value="${item.family_internal}">${esc(item.family_name)} · ${item.unique_parents} foto</option>`).join('');$('familySelect').onchange=event=>loadFamily(event.target.value).catch(error=>toast(error.message));$('photoUpload').onchange=event=>{const file=event.target.files[0];if(file)matchUpload(file).catch(error=>{$('sourceMessage').textContent=error.message;$('sourceMessage').classList.remove('hidden');toast(error.message)})};$('addCandidate').onclick=addSelectedCandidate;$('createGroup').onclick=()=>createDossier().catch(error=>toast(error.message));await resumeInspection();
}
window.addEventListener('resize',()=>alignBoxes());init().catch(error=>toast(error.message));
request('/api/collection/assembly-priorities').then(queue=>{
  if(!queue.items.length)return;
  $('assemblyQueue').classList.remove('hidden');
  $('assemblyQueue').onclick=()=>{
    $('sourceMessage').textContent=`Urutan ${queue.items.length} foto evaluasi menurut konteks perakitan. Skor untuk prioritas pemeriksaan; keputusan ditetapkan dari foto asli.`;
    $('sourceMessage').classList.remove('hidden');
    $('exampleList').innerHTML=queue.items.map(item=>`<button class="example" data-priority="${esc(item.canonical_id)}"><figure class="evidence-figure compact"><img src="${imageUrl(item.source_relative_path)}" alt="Foto prioritas ${item.rank}"></figure><span><strong>Urutan ${item.rank}</strong><small>Skor konteks ${item.score.toFixed(3)}</small></span></button>`).join('');
    document.querySelectorAll('[data-priority]').forEach(button=>button.onclick=async()=>{
      const item=queue.items.find(x=>x.canonical_id===button.dataset.priority);
      try {const photo=await request(`/api/collection/photo?sha256=${item.source_sha256}`);renderExamples(photo.items);await selectQuery(photo.items[0].region_id);}
      catch(error){$('compareEmpty').innerHTML=`<img src="${imageUrl(item.source_relative_path)}" alt="Foto utuh untuk inspeksi" style="max-width:100%;max-height:400px"><p>Foto tersedia; tidak ada region aktif untuk dossier bagian.</p>`;$('compareEmpty').classList.remove('hidden');$('comparison').classList.add('hidden');}
    });
  };
}).catch(()=>{});


let inspectionSession=null,activePhoto=null,inspectionMode=true;
const post=(url,body)=>request(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function mode(inspection){
 inspectionMode=inspection;$('inspectionTools').classList.toggle('hidden',!inspection);$('exploreTools').classList.toggle('hidden',inspection);
 $('inspectionMode').classList.toggle('active',inspection);$('exploreMode').classList.toggle('active',!inspection);
 $('photoReview').classList.toggle('hidden',!inspection||!activePhoto);$('reviewToolbar').classList.toggle('hidden',!inspection||!activePhoto);
 if(inspection)renderInspection();else loadFamily($('familySelect').value).catch(e=>toast(e.message));
}
async function resumeInspection(){inspectionSession=await request('/api/inspection/session');renderInspection();if(inspectionSession){$('inspectionBudget').value=inspectionSession.budget;const next=inspectionSession.plan.items.find(x=>!inspectionSession.reviews[x.canonical_id]);if(next)await openPlanPhoto(next.canonical_id);}}
function renderInspection(){
 if(!inspectionSession){$('exampleList').innerHTML='';return;}
 const {plan,reviews,batch_number}=inspectionSession,done=plan.items.filter(x=>reviews[x.canonical_id]).length;
 $('sessionProgress').textContent=`Batch ${batch_number} · ${done}/${plan.items.length} foto diperiksa`;
 $('sourceMessage').textContent=`Rencana: ${plan.after.families_seen} keluarga tercakup. ${plan.before.families_seen} telah hadir pada foto yang diperiksa.`;$('sourceMessage').classList.remove('hidden');
 $('nextInspection').classList.toggle('hidden',done!==plan.items.length||!plan.items.length);
 $('exampleList').innerHTML=plan.items.map(item=>`<button class="example ${reviews[item.canonical_id]?'completed':''} ${activePhoto?.canonical_id===item.canonical_id?'selected':''}" data-plan-photo="${esc(item.canonical_id)}"><figure class="evidence-figure compact"><img src="${imageUrl(item.source_relative_path)}" alt="Foto ${item.rank}"></figure><span><strong>${item.rank}. ${esc(item.global_context_name)}</strong><small>${reviews[item.canonical_id]?'✓ Diperiksa':item.added_family_ids.length+' pola untuk ditinjau'}</small></span></button>`).join('');
 document.querySelectorAll('[data-plan-photo]').forEach(button=>button.onclick=()=>openPlanPhoto(button.dataset.planPhoto).catch(e=>toast(e.message)));
 $('planExports').classList.remove('hidden');
 const exportBase=`/api/inspection/${encodeURIComponent(inspectionSession.session_id)}/export?format=`;
 $('planExports').innerHTML=`<p>Unduh hasil pemeriksaan</p><a href="${exportBase}csv">CSV</a> · <a href="${exportBase}json">JSON + bukti region</a>`;
}
async function openPlanPhoto(id){
 activePhoto=inspectionSession.plan.items.find(x=>x.canonical_id===id);if(!activePhoto)return;
 const photo=await request(`/api/collection/photo?sha256=${activePhoto.source_sha256}`);
 const contributing=photo.items.filter(x=>activePhoto.added_family_ids.includes(x.family_internal));
 const choices=contributing.length?contributing:photo.items;
 $('contributingRegion').innerHTML=choices.map(x=>`<option value="${x.region_id}">${esc(x.family_name)}</option>`).join('');
 $('contributingRegion').onchange=e=>selectQuery(e.target.value).catch(e=>toast(e.message));
 await selectQuery(choices[0].region_id);
 $('workbenchTitle').textContent=activePhoto.global_context_name;$('photoPosition').textContent=`${activePhoto.rank} / ${inspectionSession.plan.items.length}`;
 $('openOriginal').href=imageUrl(activePhoto.source_relative_path);
 const previous=inspectionSession.reviews[id];$('photoOutcome').value=previous?.outcome||'recorded';$('photoNote').value=previous?.note||'';
 $('reviewToolbar').classList.remove('hidden');$('photoReview').classList.remove('hidden');renderInspection();
}
$('inspectionMode').onclick=()=>mode(true);$('exploreMode').onclick=()=>mode(false);
$('planInspection').onclick=async()=>{try{inspectionSession=await post('/api/inspection/session',{budget:Number($('inspectionBudget').value)});activePhoto=null;mode(true);if(inspectionSession.plan.items.length)await openPlanPhoto(inspectionSession.plan.items[0].canonical_id)}catch(e){toast(e.message)}};
$('completePhoto').onclick=async()=>{if(!activePhoto)return;try{inspectionSession=await post(`/api/inspection/${inspectionSession.session_id}/review`,{version:inspectionSession.version,photo_id:activePhoto.canonical_id,outcome:$('photoOutcome').value,note:$('photoNote').value});renderInspection();const next=inspectionSession.plan.items.find(x=>!inspectionSession.reviews[x.canonical_id]);if(next)await openPlanPhoto(next.canonical_id);else toast('Seluruh foto dalam batch telah diperiksa')}catch(e){toast(e.message)}};
$('nextInspection').onclick=async()=>{try{inspectionSession=await post(`/api/inspection/${inspectionSession.session_id}/next`,{version:inspectionSession.version});activePhoto=null;renderInspection();if(inspectionSession.plan.items.length)await openPlanPhoto(inspectionSession.plan.items[0].canonical_id)}catch(e){toast(e.message)}};
