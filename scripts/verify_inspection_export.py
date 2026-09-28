"""Check session exports against the complete frozen discovery catalog."""
from pathlib import Path
import csv
import hashlib
import io
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution.evidence_intake.inspection import InspectionSessions, inspection_csv
from solution.evidence_intake.planner import load_policy


def main():
    source = ROOT/'solution/evidence_intake/demo/collection_catalog.json'
    catalog = json.loads(source.read_text())
    weights = load_policy(ROOT/'experiments/final_study_20260928/region_family_census_r2/family_census.csv')
    native = {r['region_id']: r for r in catalog['items']}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'session.db'
        store = InspectionSessions(path, catalog, weights)
        state = store.create(5)
        first = [r['canonical_id'] for r in state['plan']['items']]
        outcomes = ['recorded', 'excluded', 'uncertain', 'recorded', 'recorded']
        for photo, outcome in zip(first, outcomes):
            state = store.review(state['session_id'], state['version'], photo, outcome,
                                 'Skenario uji ekspor, bukan label komponen')
        state = store.next_batch(state['session_id'], state['version'])
        export = InspectionSessions(path, catalog, weights).export(state['session_id'])
        assert store.get(state['session_id']) == state
        assert export['counts'] == {'recorded': 3, 'excluded': 1, 'uncertain': 1, 'pending': 5}
        rows = list(csv.DictReader(io.StringIO(inspection_csv(export).decode('utf-8-sig'))))
        assert len(rows) == 10 and len({r['photo_id'] for r in rows}) == 10
        regions = 0
        for photo in export['photos']:
            for region in photo['region_evidence']:
                ref = native[region['region_id']]
                assert ref['parent_id'] == photo['photo_id']
                assert ref['source_sha256'] == photo['source_sha256']
                assert ref['source_relative_path'] == photo['source_relative_path']
                for key in ['bbox_xyxy', 'mask_sha256', 'family_internal', 'width', 'height']:
                    assert region[key] == ref[key]
                regions += 1
    result = {'status': 'pass', 'catalog_regions': len(native), 'exported_photos': len(rows),
              'verified_region_records': regions, 'outcomes': export['counts'],
              'resume_preserves_saved_state': True, 'export_is_read_only': True,
              'catalog_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'scope': 'Scripted functional check over the frozen catalog, not operator or semantic validation.'}
    out = ROOT/'experiments/final_study_20260928/inspection_worklist_r1/export_receipt.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
