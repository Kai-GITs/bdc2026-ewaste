"""Exercise persistent inspection scheduling, nonrepeat replanning and exports."""
import argparse,json,hashlib
from pathlib import Path
from urllib.request import Request,urlopen

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',default='http://127.0.0.1:8879');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    def call(path,body=None):
        req=Request(a.base_url+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=30) as r:return r.read()
    one=json.loads(call('/api/collection/inspection-plan',{'budget':25,'save':True,'batch_id':'INSPEKSI-SD2026040000363'}))
    reviewed=[r['canonical_id'] for r in one['items']]
    assert len(reviewed)==25 and len(set(reviewed))==25
    two=json.loads(call('/api/collection/inspection-plan',{'budget':25,'reviewed_photo_ids':reviewed,'save':True,'batch_id':'INSPEKSI-LANJUTAN-SD2026040000363'}))
    assert len(two['items'])==25 and not set(reviewed)&{r['canonical_id'] for r in two['items']}
    assert two['before']==one['after']
    ident=one['dossier']['dossier_id'];stored=json.loads(call('/api/dossiers/'+ident))
    assert stored['state']['inspection_plan']['items']==one['items']
    exports={}
    for ext in ['json','csv','pdf']:
        b=call('/api/dossiers/'+ident+'/export?format='+ext)
        if ext=='pdf':assert b.startswith(b'%PDF')
        if ext=='json':assert len(json.loads(b)['entries'])>0
        path=a.output/f'inspection_export.{ext}';path.write_bytes(b)
        exports[ext]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    receipt={'status':'pass','first_batch':one['after'],'second_batch':two['after'],'distinct_photos_across_batches':50,'persistent_plan':True,'exports':exports,'scope':'known collection; HTTP functionality, not operator time or physical sorting validation'}
    (a.output/'functional_test_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
