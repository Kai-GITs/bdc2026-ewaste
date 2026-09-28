import csv
import tempfile
import unittest
from pathlib import Path
from solution.evidence_intake.priorities import load_priorities

class PriorityContract(unittest.TestCase):
    def test_rank_does_not_depend_on_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.csv'
            rows=[{'canonical_id':str(i),'source_sha256':'a'*64,'source_relative_path':f'{i}.jpg',
                   'score_global_dinov3':score,'target':label,'label':label}
                  for i,score,label in [(1,.2,1),(2,.8,0)]]
            def write():
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            write();before=load_priorities(p)
            for row in rows:row['target']=1-int(row['target']);row['label']='corrupted'
            write();after=load_priorities(p)
            self.assertEqual(before,after)
            self.assertEqual(before['items'][0]['canonical_id'],'2')
            self.assertTrue(all('target' not in item and 'label' not in item for item in before['items']))
    def test_missing_configuration(self):
        self.assertEqual(load_priorities(None)['items'],[])

if __name__=='__main__':unittest.main()
