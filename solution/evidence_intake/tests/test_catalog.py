import json

import pytest

from solution.evidence_intake.catalog import (
    build_batch_from_regions,
    family_items,
    load_catalog,
    photo_items_by_sha,
    query_region,
)
from solution.evidence_intake.core import build_dossier


def sample_catalog():
    base = {
        "family_internal": 41, "machine_family_id": "F0041", "family_key": "ic",
        "family_name": "kemasan IC", "scientific_role": "cross_context_part",
        "r3_decision": "retained", "source_relative_path": "1_Electronic/a.jpg",
        "source_sha256": "a" * 64, "width": 100, "height": 80,
        "bbox_xyxy": [1, 2, 30, 40], "mask_sha256": "b" * 64,
        "area_fraction": .1, "family_affinity": .8,
        "global_community": 1, "global_context_name": "Ponsel",
    }
    a = {**base, "region_id": "r1", "parent_id": "p1", "is_r3_core": True,
         "neighbors": [{"region_id": "r2", "similarity": .9}],
         "candidates": [{"region_id": "r2", "family_internal": 41,
                         "similarity": .9, "same_family": True}]}
    b = {**base, "region_id": "r2", "parent_id": "p2", "source_sha256": "c" * 64,
         "is_r3_core": False,
         "neighbors": [{"region_id": "r1", "similarity": .9}]}
    return {
        "schema_version": 1, "scope": "test", "summary": {}, "limitations": [],
        "revisions": {"r2_partition": "r2", "r3_adjudication": "r3"},
        "families": [{"family_internal": 41, "family_key": "ic", "unique_parents": 2}],
        "items": [a, b],
    }


def test_catalog_load_query_and_family_page(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(sample_catalog()), encoding="utf-8")
    catalog = load_catalog(path)
    page = family_items(catalog, "ic", limit=1)
    assert page["total"] == 2
    assert page["items"][0]["region_id"] == "r1"
    result = query_region(catalog, "r1")
    assert result["neighbors"][0]["parent_id"] == "p2"
    assert result["candidates"][0]["same_global_context"] is True
    exact = photo_items_by_sha(catalog, "a" * 64)
    assert exact["new_photo_inference_performed"] is False
    assert exact["items"][0]["region_id"] == "r1"


def test_selection_builds_a_real_review_dossier():
    batch = build_batch_from_regions(sample_catalog(), ["r1", "r2"], "selected")
    state = build_dossier(batch)
    assert state["photos_received"] == 2
    assert len(state["entries"]) == 2
    assert {row["support_status"] for row in state["entries"]} == {"r3_core", "r2_discovery_candidate"}
    assert all(row["evidence_status"] == "model_predicted_visible" for row in state["entries"])


def test_empty_or_unknown_selection_is_rejected():
    with pytest.raises(ValueError):
        build_batch_from_regions(sample_catalog(), [], "empty")
    with pytest.raises(KeyError):
        build_batch_from_regions(sample_catalog(), ["missing"], "bad")
