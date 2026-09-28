"""Persistent, versioned per-photo inspection sessions over a discovery catalog."""
import json
import csv
import io
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .planner import plan_inspection, photo_index


class InspectionSessions:
    def __init__(self, path, catalog, weights=None):
        self.path = Path(path)
        self.catalog = catalog
        self.weights = weights or {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS inspection_sessions (session_id TEXT PRIMARY KEY, version INTEGER NOT NULL, updated_at TEXT NOT NULL, state_json TEXT NOT NULL)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, session_id=None):
        with self.connect() as db:
            row = db.execute('SELECT state_json FROM inspection_sessions WHERE session_id=?', (session_id,)).fetchone() if session_id else db.execute('SELECT state_json FROM inspection_sessions ORDER BY updated_at DESC LIMIT 1').fetchone()
        if row is None:
            if session_id: raise KeyError('Inspection session not found')
            return None
        return json.loads(row[0])

    def create(self, budget):
        plan = plan_inspection(self.catalog, budget, self.weights)
        state = {'session_id':uuid.uuid4().hex, 'version':1, 'budget':int(budget), 'reviews':{}, 'batch_number':1, 'plan':plan}
        with self.connect() as db:
            db.execute('INSERT INTO inspection_sessions VALUES (?,?,?,?)', (state['session_id'],1,datetime.now(timezone.utc).isoformat(),json.dumps(state)))
        return state

    def _update(self, session_id, version, mutation):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT version,state_json FROM inspection_sessions WHERE session_id=?',(session_id,)).fetchone()
            if row is None: raise KeyError('Inspection session not found')
            if row[0] != int(version): raise ValueError('Session changed. Reload before saving this review.')
            state=json.loads(row[1]);mutation(state);state['version']+=1
            db.execute('UPDATE inspection_sessions SET version=?,updated_at=?,state_json=? WHERE session_id=?',(state['version'],datetime.now(timezone.utc).isoformat(),json.dumps(state),session_id))
        return state

    def review(self, session_id, version, photo_id, outcome, note=''):
        if outcome not in {'recorded','uncertain','excluded'}: raise ValueError('Invalid inspection outcome')
        if len(note)>4000: raise ValueError('Note exceeds 4000 characters')
        def mutate(state):
            if photo_id not in {r['canonical_id'] for r in state['plan']['items']}: raise ValueError('Photo is outside this batch')
            state['reviews'][photo_id]={'outcome':outcome,'note':note,'completed_at':datetime.now(timezone.utc).isoformat()}
        return self._update(session_id,version,mutate)

    def next_batch(self, session_id, version):
        def mutate(state):
            if any(r['canonical_id'] not in state['reviews'] for r in state['plan']['items']): raise ValueError('Finish each photo in the current batch before continuing.')
            # Completed inspection does not certify family identity. Excluded photos
            # leave the candidate pool and cannot contribute coverage.
            completed={p for p,r in state['reviews'].items() if r['outcome']!='excluded'}
            excluded={p for p,r in state['reviews'].items() if r['outcome']=='excluded'}
            state['plan']=plan_inspection(self.catalog,state['budget'],self.weights,reviewed=completed,eligible=set(photo_index(self.catalog))-excluded)
            state['batch_number']+=1
        return self._update(session_id,version,mutate)

    def export(self, session_id):
        """Export a saved session with source evidence, without changing reviews.

        Family memberships describe the frozen catalog. A completed photo review
        is not a confirmation of every region or family in that photo.
        """
        state = self.get(session_id)
        photos = photo_index(self.catalog)
        regions = {r['region_id']: r for r in self.catalog['items']}
        order = list(state['reviews'])
        order.extend(r['canonical_id'] for r in state['plan']['items']
                     if r['canonical_id'] not in state['reviews'])
        rows = []
        for photo_id in order:
            source = photos[photo_id]
            review = state['reviews'].get(photo_id)
            evidence = []
            for region_id in source['regions']:
                region = regions[region_id]
                evidence.append({key: region[key] for key in (
                    'region_id', 'family_internal', 'family_key', 'bbox_xyxy',
                    'mask_rle', 'mask_path', 'mask_polygon', 'mask_sha256',
                    'width', 'height', 'is_r3_core'
                ) if key in region})
            rows.append({
                'photo_id': photo_id,
                'source_relative_path': source['source_relative_path'],
                'source_sha256': source['source_sha256'],
                'global_context_name': source['global_context_name'],
                'outcome': review['outcome'] if review else 'pending',
                'note': review['note'] if review else '',
                'completed_at': review['completed_at'] if review else None,
                'family_ids': sorted(source['families']),
                'region_evidence': evidence,
            })
        return {
            'schema_version': 1,
            'session_id': state['session_id'], 'version': state['version'],
            'batch_number': state['batch_number'],
            'counts': {outcome: sum(r['outcome'] == outcome for r in rows)
                       for outcome in ('recorded', 'uncertain', 'excluded', 'pending')},
            'evidence_scope': 'Catalog memberships are model outputs. Review outcomes apply to photos, not to each component identity.',
            'photos': rows,
        }


def inspection_csv(export):
    """One row per photo, including pending and excluded records."""
    stream = io.StringIO(newline='')
    fields = ['photo_id', 'source_relative_path', 'source_sha256',
              'global_context_name', 'outcome', 'note', 'completed_at',
              'family_ids', 'region_ids']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in export['photos']:
        flat = {k: row[k] for k in fields if k in row}
        flat['family_ids'] = ';'.join(map(str, row['family_ids']))
        flat['region_ids'] = ';'.join(r['region_id'] for r in row['region_evidence'])
        # Preserve notes as text when this CSV is opened in spreadsheet software.
        for key, value in flat.items():
            if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
                flat[key] = "'" + value
        writer.writerow(flat)
    return ('\ufeff' + stream.getvalue()).encode('utf-8')
