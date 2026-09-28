import json

import pytest

from solution.evidence_intake.core import (
    DossierStore,
    build_dossier,
    dossier_csv,
    dossier_pdf,
)


def batch():
    duplicate_sha = "a" * 64
    return {
        "batch_id": "uji-001",
        "discovery_revision": "test-revision",
        "photos": [
            {"photo_id": "p1", "sha256": duplicate_sha, "readable": True},
            {"photo_id": "p1-copy", "sha256": duplicate_sha, "readable": True},
            {"photo_id": "p2", "sha256": "b" * 64, "readable": False},
            {"photo_id": "p3", "sha256": "c" * 64, "readable": True,
             "closed_casing": True, "sensor_ambiguity_cue": True},
        ],
        "family_support": [
            {"photo_id": "p1", "family_key": "f7", "family_name": "Kipas terlihat",
             "region_id": "r1", "support_strength": 0.8, "bbox_xyxy": [1, 2, 30, 40]},
        ],
        "ownership_evidence": [
            {"photo_id": "p1", "region_id": "r1",
             "ownership_status": "geometric_parent_candidate", "owner_region_id": "r-parent"},
        ],
    }


def test_empty_batch_is_valid_and_explicit():
    result = build_dossier({"batch_id": "empty", "photos": []})
    assert result["photos_received"] == 0
    assert result["entries"] == []


def test_dedup_unreadable_unsupported_and_hidden_guard():
    result = build_dossier(batch())
    assert result["photos_received"] == 4
    assert result["unique_image_records"] == 3
    assert result["duplicate_groups"][0]["photo_ids"] == ["p1", "p1-copy"]
    assert len(result["entries"]) == 1
    assert result["entries"][0]["hidden_content_claim"] is False
    reasons = {row["reason"] for row in result["evidence_requests"]}
    assert "unreadable_image" in reasons
    assert "unsupported_or_no_visible_family" in reasons
    assert "closed_casing_with_reference_ambiguity" in reasons
    assert all(not row["requires_disassembly"] for row in result["evidence_requests"])


def test_revision_reopen_and_reexport(tmp_path):
    store = DossierStore(tmp_path / "dossiers.sqlite3")
    initial = store.create(build_dossier(batch()))
    entry_id = initial["state"]["entries"][0]["entry_id"]
    revised = store.revise(initial["dossier_id"], [{
        "entry_id": entry_id,
        "changes": {"visible_part_label": "Kipas terkonfirmasi", "evidence_status": "observed_manual",
                    "review_status": "accepted"},
        "reason": "visual review",
    }])
    assert revised["revision"] == 2
    assert store.get(initial["dossier_id"], 1)["state"]["entries"][0]["visible_part_label"] == "Kipas terlihat"
    assert store.get(initial["dossier_id"])["state"]["entries"][0]["visible_part_label"] == "Kipas terkonfirmasi"
    assert dossier_csv(revised["state"]).startswith(b"\xef\xbb\xbf")
    pdf = dossier_pdf(revised["state"])
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000


def test_invalid_hidden_or_ownership_correction_is_rejected(tmp_path):
    store = DossierStore(tmp_path / "dossiers.sqlite3")
    initial = store.create(build_dossier(batch()))
    entry_id = initial["state"]["entries"][0]["entry_id"]
    with pytest.raises(ValueError):
        store.revise(initial["dossier_id"], [{"entry_id": entry_id, "changes": {"hidden_content_claim": True}}])
    with pytest.raises(ValueError):
        store.revise(initial["dossier_id"], [{"entry_id": entry_id, "changes": {"ownership_status": "definitely_attached"}}])
