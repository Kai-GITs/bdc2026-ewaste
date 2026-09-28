"""A context-based review order over a frozen evaluated cohort.

Only identifiers, provenance and prediction scores enter this interface. Evaluation
labels are deliberately excluded from its output and have no effect on ranking.
"""
import csv
import math
from pathlib import Path

def load_priorities(path: Path | None) -> dict:
    if path is None:
        return {"scope": "not_configured", "items": []}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    items=[]
    seen=set()
    for row in rows:
        ident=row['canonical_id']
        if ident in seen:
            raise ValueError('Repeated canonical photo in priority cohort')
        seen.add(ident)
        score=float(row['score_global_dinov3'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid context score')
        items.append({'canonical_id':ident,'source_sha256':row['source_sha256'],
                      'source_relative_path':row['source_relative_path'],'score':score})
    items.sort(key=lambda item:(-item['score'],item['canonical_id']))
    for rank,item in enumerate(items,1):item['rank']=rank
    return {'scope':'frozen_evaluated_cohort','score_meaning':'ranking score, not calibrated probability',
            'new_photo_inference':False,'items':items}
