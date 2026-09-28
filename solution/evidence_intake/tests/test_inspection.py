import pytest
from solution.evidence_intake.inspection import InspectionSessions
from test_planner import catalog


def test_resume_partial_batch_and_no_bulk_completion(tmp_path):
    path=tmp_path/'inspection.sqlite3'
    store=InspectionSessions(path,catalog());s=store.create(2)
    a,b=[r['canonical_id'] for r in s['plan']['items']]
    s=store.review(s['session_id'],s['version'],a,'recorded','visible pattern recorded')
    resumed=InspectionSessions(path,catalog()).get()
    assert resumed==s and b not in resumed['reviews']
    with pytest.raises(ValueError,match='Finish each'): store.next_batch(s['session_id'],s['version'])
    assert store.get()['version']==s['version']
    s=store.review(s['session_id'],s['version'],b,'uncertain')
    s=store.next_batch(s['session_id'],s['version'])
    assert not {a,b}&{r['canonical_id'] for r in s['plan']['items']}


def test_stale_review_and_excluded_photo_do_not_change_coverage(tmp_path):
    store=InspectionSessions(tmp_path/'state.db',catalog());s=store.create(1)
    p=s['plan']['items'][0]['canonical_id'];old=s['version']
    s=store.review(s['session_id'],old,p,'excluded','not suitable')
    with pytest.raises(ValueError,match='Session changed'):store.review(s['session_id'],old,p,'recorded')
    with pytest.raises(ValueError,match='outside'):store.review(s['session_id'],s['version'],'missing','recorded')
    s=store.next_batch(s['session_id'],s['version'])
    assert s['plan']['before']=={'families_seen':0,'families_repeated':0}
    assert p not in {r['canonical_id'] for r in s['plan']['items']}


def test_export_preserves_pending_excluded_and_source_after_resume(tmp_path):
    import csv, io, json
    from solution.evidence_intake.inspection import inspection_csv
    path = tmp_path/'export.db'
    data = catalog()
    data['items'][0].update(bbox_xyxy=[0, 1, 10, 20], mask_sha256='f'*64)
    store = InspectionSessions(path, data)
    state = store.create(2)
    first, second = [r['canonical_id'] for r in state['plan']['items']]
    state = store.review(state['session_id'], state['version'], first, 'excluded', 'Papan, kabel\nPerlu foto lain')
    export = store.export(state['session_id'])
    assert export['counts'] == {'recorded': 0, 'uncertain': 0, 'excluded': 1, 'pending': 1}
    indexed = {r['photo_id']: r for r in export['photos']}
    assert indexed[second]['completed_at'] is None
    assert indexed[first]['source_relative_path'] == first+'.jpg'
    assert indexed[first]['source_sha256'] == first*64
    assert indexed['a']['region_evidence'][0]['mask_sha256'] == 'f'*64
    assert json.loads(json.dumps(export)) == InspectionSessions(path, data).export(state['session_id'])
    rows = list(csv.DictReader(io.StringIO(inspection_csv(export).decode('utf-8-sig'))))
    assert rows[0]['note'] == 'Papan, kabel\nPerlu foto lain'
    assert store.get(state['session_id']) == state
    state = store.review(state['session_id'], state['version'], second, 'uncertain', '=1+1')
    state = store.next_batch(state['session_id'], state['version'])
    export = store.export(state['session_id'])
    assert len({r['photo_id'] for r in export['photos']}) == len(export['photos'])
    assert {r['photo_id'] for r in export['photos']} >= {first, second}
    rows = list(csv.DictReader(io.StringIO(inspection_csv(export).decode('utf-8-sig'))))
    assert next(r for r in rows if r['photo_id'] == second)['note'] == "'=1+1"
    assert next(r for r in export['photos'] if r['photo_id'] == second)['note'] == '=1+1'


def test_saved_session_export_http(tmp_path):
    import csv, io, json, threading
    from urllib.request import urlopen
    from urllib.error import HTTPError
    from http.server import ThreadingHTTPServer
    from solution.evidence_intake.server import WorkspaceHandler
    store = InspectionSessions(tmp_path/'http.db', catalog())
    state = store.create(2)
    class Handler(WorkspaceHandler):
        sessions = store
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}/api/inspection/{state["session_id"]}/export?format='
    try:
        with urlopen(base+'json') as response:
            assert response.headers['Content-Type'].startswith('application/json')
            assert 'attachment;' in response.headers['Content-Disposition']
            assert json.load(response) == store.export(state['session_id'])
        with urlopen(base+'csv') as response:
            rows = list(csv.DictReader(io.StringIO(response.read().decode('utf-8-sig'))))
            assert len(rows) == 2 and all(r['outcome'] == 'pending' for r in rows)
        with pytest.raises(HTTPError) as error: urlopen(base+'pdf')
        assert error.value.code == 400
    finally:
        server.shutdown(); server.server_close(); worker.join()
