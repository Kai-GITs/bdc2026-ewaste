"""Deterministic inspection scheduling from a frozen discovery catalog.

The planner sees r2 memberships and an explicit family-utility policy. It does
not read r3 support flags, relation labels, or evaluation targets. Its coverage
objective is monotone submodular with a photo-count budget.
"""
import csv
import math
from pathlib import Path


def load_policy(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        weights = {}
        for row in csv.DictReader(f):
            coherence = float(row['coherence_0_4'])
            shortcut = float(row['shortcut_risk_0_4'])
            if not 0 <= coherence <= 4 or not 0 <= shortcut <= 4:
                raise ValueError('Policy ratings must be within 0..4')
            weights[int(row['family_internal'])] = coherence / 4 * (1 - shortcut / 4)
    return weights


def photo_index(catalog):
    photos = {}
    for row in catalog['items']:
        parent = row['parent_id']
        if parent not in photos:
            photos[parent] = {'canonical_id': parent, 'source_relative_path': row['source_relative_path'],
                             'source_sha256': row['source_sha256'], 'global_community': row['global_community'],
                             'global_context_name': row['global_context_name'], 'families': set(), 'regions': []}
        photos[parent]['families'].add(int(row['family_internal']))
        photos[parent]['regions'].append(row['region_id'])
    return photos


def plan_inspection(catalog, budget=25, weights=None, reviewed=None, eligible=None, family_ids=None, repeats=2):
    budget = int(budget)
    repeats = int(repeats)
    if not 1 <= budget <= 500 or not 1 <= repeats <= 5:
        raise ValueError('Budget must be 1..500 and distinct-photo target 1..5')
    photos = photo_index(catalog)
    all_families = {int(f['family_internal']) for f in catalog['families']}
    requested = set(map(int, family_ids)) if family_ids is not None else all_families
    if not requested <= all_families:
        raise ValueError('Unknown family requested')
    weights = {f: float((weights or {}).get(f, 1.0)) for f in requested}
    if any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('Weights must be finite and nonnegative')
    reviewed = set(reviewed or [])
    eligible = set(photos) if eligible is None else set(eligible)
    unknown = (reviewed | eligible) - set(photos)
    if unknown:
        raise ValueError('Photos without catalog regions cannot enter this plan')
    counts = {f: sum(f in photos[p]['families'] for p in reviewed) for f in requested}
    initial = dict(counts)
    candidates = sorted(eligible - reviewed)
    chosen = []
    for step in range(min(budget, len(candidates))):
        scored = [(sum(weights.get(f, 0) / repeats for f in photos[p]['families'] if counts.get(f, repeats) < repeats), p) for p in candidates]
        gain, parent = min(scored, key=lambda pair: (-pair[0], pair[1]))
        if gain <= 0:
            break
        useful = sorted(f for f in photos[parent]['families'] if weights.get(f, 0) > 0 and counts.get(f, repeats) < repeats)
        item = {k:v for k,v in photos[parent].items() if k not in ['families','regions']}
        item.update(rank=len(chosen)+1, marginal_gain=gain, added_family_ids=useful,
                    region_ids=photos[parent]['regions'])
        chosen.append(item)
        for f in photos[parent]['families'] & requested: counts[f] += 1
        candidates.remove(parent)
    return {'scope':'inspection scheduling over existing discovery records',
            'requested_budget':budget,'eligible_photos':len(eligible),'reviewed_photos':len(reviewed),
            'target_distinct_photos_per_family':repeats,'policy':'weighted capped family coverage',
            'used_r3_or_evaluation_labels':False,'items':chosen,
            'before':{'families_seen':sum(v>0 for v in initial.values()),'families_repeated':sum(v>=repeats for v in initial.values())},
            'after':{'families_seen':sum(v>0 for v in counts.values()),'families_repeated':sum(v>=repeats for v in counts.values())},
            'objective':sum(weights[f]*min(counts[f],repeats)/repeats for f in requested)}
