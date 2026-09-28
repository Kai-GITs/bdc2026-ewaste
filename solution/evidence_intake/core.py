"""Build and revise evidence-first e-waste intake dossiers.

The engine consumes research outputs; it does not infer hidden components.  A
region-family assignment is recorded as model-predicted visible evidence until
a reviewer explicitly confirms or corrects it.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EVIDENCE_STATUSES = {
    "observed_manual",
    "model_predicted_visible",
    "unresolved",
    "unsupported",
}
OWNERSHIP_STATUSES = {
    "same_unit_confirmed_manual",
    "geometric_parent_candidate",
    "same_scene",
    "unresolved",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _validate_batch(batch: dict[str, Any]) -> None:
    if not isinstance(batch.get("batch_id"), str) or not batch["batch_id"].strip():
        raise ValueError("batch_id must be a non-empty string")
    photos = batch.get("photos")
    if not isinstance(photos, list):
        raise ValueError("photos must be a list")
    ids = [row.get("photo_id") for row in photos]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every photo needs a photo_id")
    if len(ids) != len(set(ids)):
        raise ValueError("photo_id values must be unique within a batch")


def _deduplicate(photos: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Consolidate exact byte duplicates and retain every original identifier."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for photo in photos:
        sha = str(photo.get("sha256") or "").strip().lower()
        key = f"sha256:{sha}" if len(sha) == 64 else f"photo:{photo['photo_id']}"
        groups[key].append(photo)

    unique: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    for key, members in groups.items():
        representative = copy.deepcopy(members[0])
        representative["source_photo_ids"] = [row["photo_id"] for row in members]
        representative["duplicate_status"] = (
            "exact_duplicate_group" if len(members) > 1 else "unique_or_unresolved"
        )
        unique.append(representative)
        if len(members) > 1:
            duplicate_groups.append({
                "kind": "exact_bytes",
                "sha256": members[0].get("sha256"),
                "representative_photo_id": members[0]["photo_id"],
                "photo_ids": [row["photo_id"] for row in members],
                "physical_unit_count": None,
                "note": "File-identical photos are one image record; this does not prove one physical unit.",
            })
    return unique, duplicate_groups


def _index_supports(supports: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in supports:
        photo_id = row.get("photo_id")
        if isinstance(photo_id, str):
            indexed[photo_id].append(copy.deepcopy(row))
    return indexed


def build_dossier(batch: dict[str, Any]) -> dict[str, Any]:
    """Create a traceable draft from photos and precomputed research evidence."""
    _validate_batch(batch)
    photos, duplicate_groups = _deduplicate(batch["photos"])
    supports = _index_supports(batch.get("family_support", []))
    ownership = {
        (row.get("photo_id"), row.get("region_id")): row
        for row in batch.get("ownership_evidence", [])
    }
    requests: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    for photo in photos:
        photo_id = photo["photo_id"]
        readable = bool(photo.get("readable", True))
        if not readable:
            requests.append({
                "photo_id": photo_id,
                "reason": "unreadable_image",
                "request": "Kirim ulang foto yang fokus dan cukup terang dari sudut yang sama.",
                "requires_disassembly": False,
            })
            continue

        photo_supports: list[dict[str, Any]] = []
        for source_id in photo["source_photo_ids"]:
            photo_supports.extend(supports.get(source_id, []))
        if not photo_supports:
            requests.append({
                "photo_id": photo_id,
                "reason": "unsupported_or_no_visible_family",
                "request": "Tambahkan foto utuh dan foto dekat bagian yang sudah terlihat dari sudut lain.",
                "requires_disassembly": False,
            })

        seen_family: set[str] = set()
        for support in sorted(
            photo_supports,
            key=lambda row: (-float(row.get("support_strength") or 0.0), str(row.get("family_key"))),
        ):
            family_key = str(support.get("family_key") or "unresolved")
            if family_key in seen_family:
                continue
            seen_family.add(family_key)
            region_id = str(support.get("region_id") or "")
            relation = ownership.get((support.get("photo_id"), region_id), {})
            owner_status = str(relation.get("ownership_status") or "unresolved")
            if owner_status not in OWNERSHIP_STATUSES:
                owner_status = "unresolved"
            label = str(support.get("family_name") or "Keluarga visual belum ditafsirkan")
            entries.append({
                "entry_id": _stable_id("entry"),
                "photo_id": photo_id,
                "source_photo_ids": photo["source_photo_ids"],
                "family_key": family_key,
                "visible_part_label": label,
                "region_id": region_id,
                "bbox_xyxy": support.get("bbox_xyxy"),
                "evidence_status": "model_predicted_visible",
                "support_strength": support.get("support_strength"),
                "support_status": support.get("support_status", "unresolved"),
                "claim_gate_status": support.get("claim_gate_status", "needs_visual_review"),
                "machine_family_id": support.get("machine_family_id"),
                "mask_sha256": support.get("mask_sha256"),
                "ownership_status": owner_status,
                "owner_region_id": relation.get("owner_region_id"),
                "hidden_content_claim": False,
                "review_status": "needs_review",
                "review_note": "",
            })

        if photo.get("sensor_ambiguity_cue"):
            ambiguity = str(photo.get("ambiguity_kind") or (
                "closed_casing_with_reference_ambiguity" if photo.get("closed_casing") else "reference_ambiguity"
            ))
            requests_by_kind = {
                "display_state": "Tambahkan foto layar saat perangkat dinyalakan atau dari sudut lain, tanpa pembongkaran.",
                "attachment_context": "Tambahkan foto utuh dan foto samping yang memperlihatkan hubungan bagian dengan perangkat.",
                "reference_ambiguity": "Jika tersedia tanpa pembongkaran baru, tambahkan foto kompartemen atau unit yang sudah terbuka.",
                "closed_casing_with_reference_ambiguity": "Jika tersedia tanpa pembongkaran baru, tambahkan foto kompartemen atau unit yang sudah terbuka.",
            }
            requests.append({
                "photo_id": photo_id,
                "reason": ambiguity,
                "request": requests_by_kind.get(ambiguity, requests_by_kind["reference_ambiguity"]),
                "requires_disassembly": False,
                "sensor_role": "request_additional_evidence_only",
            })

    return {
        "schema_version": 1,
        "dossier_id": str(batch.get("dossier_id") or _stable_id("dossier")),
        "batch_id": batch["batch_id"],
        "created_at": _now(),
        "method": {
            "discovery_revision": batch.get("discovery_revision"),
            "family_name_map_revision": batch.get("family_name_map_revision"),
            "rule": "Only visible region evidence creates entries; no hidden component is inferred.",
        },
        "photos_received": len(batch["photos"]),
        "unique_image_records": len(photos),
        "photos": photos,
        "duplicate_groups": duplicate_groups,
        "entries": entries,
        "evidence_requests": requests,
        "review": {
            "status": "draft",
            "physical_unit_count": None,
            "approved_by": None,
            "approved_at": None,
        },
        "claims_excluded": [
            "identitas komponen tersembunyi",
            "jumlah unit fisik dari jumlah foto",
            "komposisi material atau nilai",
            "toksisitas, keselamatan, atau instruksi pembongkaran",
        ],
    }


def apply_corrections(state: dict[str, Any], corrections: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply reviewer edits without changing discovery assignments."""
    revised = copy.deepcopy(state)
    entry_by_id = {row["entry_id"]: row for row in revised.get("entries", [])}
    audit = revised.setdefault("correction_log", [])
    allowed = {
        "visible_part_label",
        "evidence_status",
        "ownership_status",
        "owner_region_id",
        "review_status",
        "review_note",
    }
    for correction in corrections:
        entry_id = correction.get("entry_id")
        if entry_id not in entry_by_id:
            raise KeyError(f"unknown entry_id: {entry_id}")
        changes = correction.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise ValueError("each correction needs non-empty changes")
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"fields cannot be corrected: {sorted(unknown)}")
        if "evidence_status" in changes and changes["evidence_status"] not in EVIDENCE_STATUSES:
            raise ValueError("invalid evidence_status")
        if "ownership_status" in changes and changes["ownership_status"] not in OWNERSHIP_STATUSES:
            raise ValueError("invalid ownership_status")
        before = {key: entry_by_id[entry_id].get(key) for key in changes}
        entry_by_id[entry_id].update(changes)
        entry_by_id[entry_id]["hidden_content_claim"] = False
        audit.append({
            "entry_id": entry_id,
            "changed_at": _now(),
            "reason": str(correction.get("reason") or "reviewer correction"),
            "before": before,
            "after": changes,
        })
    revised["review"]["status"] = "revised"
    return revised


class DossierStore:
    """SQLite revision store. Every correction creates an immutable snapshot."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS revisions (
                    dossier_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    parent_revision INTEGER,
                    created_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY (dossier_id, revision)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def create(self, state: dict[str, Any], reason: str = "initial draft") -> dict[str, Any]:
        dossier_id = state["dossier_id"]
        with self._connect() as db:
            db.execute(
                "INSERT INTO revisions VALUES (?, 1, NULL, ?, ?, ?)",
                (dossier_id, _now(), reason, json.dumps(state, ensure_ascii=False)),
            )
        return self.get(dossier_id, 1)

    def revise(
        self,
        dossier_id: str,
        corrections: list[dict[str, Any]],
        reason: str = "reviewer correction",
    ) -> dict[str, Any]:
        prior = self.get(dossier_id)
        revised = apply_corrections(prior["state"], corrections)
        next_revision = prior["revision"] + 1
        with self._connect() as db:
            db.execute(
                "INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?)",
                (dossier_id, next_revision, prior["revision"], _now(), reason,
                 json.dumps(revised, ensure_ascii=False)),
            )
        return self.get(dossier_id, next_revision)

    def get(self, dossier_id: str, revision: int | None = None) -> dict[str, Any]:
        sql = (
            "SELECT revision,parent_revision,created_at,reason,state_json FROM revisions "
            "WHERE dossier_id=? "
        )
        params: list[Any] = [dossier_id]
        if revision is None:
            sql += "ORDER BY revision DESC LIMIT 1"
        else:
            sql += "AND revision=?"
            params.append(revision)
        with self._connect() as db:
            row = db.execute(sql, params).fetchone()
        if row is None:
            raise KeyError(f"dossier/revision not found: {dossier_id}/{revision}")
        return {
            "dossier_id": dossier_id,
            "revision": row[0],
            "parent_revision": row[1],
            "revision_created_at": row[2],
            "revision_reason": row[3],
            "state": json.loads(row[4]),
        }

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT dossier_id,MAX(revision) FROM revisions GROUP BY dossier_id ORDER BY dossier_id"
            ).fetchall()
        return [self.get(dossier_id, revision) for dossier_id, revision in rows]


def dossier_csv(state: dict[str, Any]) -> bytes:
    fields = [
        "entry_id", "photo_id", "visible_part_label", "family_key", "region_id",
        "evidence_status", "ownership_status", "owner_region_id", "review_status",
        "review_note", "hidden_content_claim",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(state.get("entries", []))
    return stream.getvalue().encode("utf-8-sig")


def dossier_pdf(state: dict[str, Any]) -> bytes:
    """Render a compact staff handover PDF with traceable evidence statuses."""
    import fitz

    document = fitz.open()
    page = document.new_page(width=595, height=842)
    y = 48.0

    def line(text: str, size: float = 9.2, bold: bool = False, gap: float = 15.0) -> None:
        nonlocal page, y
        if y > 790:
            page = document.new_page(width=595, height=842)
            y = 48.0
        font = "hebo" if bold else "helv"
        page.insert_text((48, y), text[:110], fontsize=size, fontname=font, color=(0.08, 0.12, 0.16))
        y += gap

    line("BERKAS PENERIMAAN E-WASTE BERBASIS BUKTI", 14, True, 23)
    line(f"Batch: {state['batch_id']}   |   Berkas: {state['dossier_id']}", 9.5, True)
    line(f"Foto diterima: {state['photos_received']}   |   Rekaman gambar unik: {state['unique_image_records']}")
    line("Draf ini hanya mencatat bagian yang mempunyai bukti visual; isi tersembunyi tidak disimpulkan.", 8.6, False, 21)
    line("ENTRI BUKTI", 10.5, True, 19)
    if not state.get("entries"):
        line("Belum ada entri terdukung.")
    for entry in state.get("entries", []):
        label = entry["visible_part_label"]
        status = {
            "observed_manual": "diamati dan dikonfirmasi",
            "model_predicted_visible": "prediksi bagian terlihat",
            "unresolved": "belum terselesaikan",
            "unsupported": "tidak terdukung",
        }.get(entry["evidence_status"], entry["evidence_status"].replace("_", " "))
        owner = {
            "same_unit_confirmed_manual": "satu unit, dikonfirmasi",
            "geometric_parent_candidate": "kandidat induk geometris",
            "same_scene": "satu adegan",
            "unresolved": "belum jelas",
        }.get(entry["ownership_status"], entry["ownership_status"].replace("_", " "))
        line(f"• {label} — {status}; relasi: {owner}", 8.8, False, 13)
        line(f"  foto {entry['photo_id']} | region {entry['region_id']}", 7.7, False, 12)
    y += 8
    line("BUKTI TAMBAHAN YANG DIMINTA", 10.5, True, 19)
    if not state.get("evidence_requests"):
        line("Tidak ada permintaan tambahan otomatis.")
    for request in state.get("evidence_requests", []):
        line(f"• {request['photo_id']}: {request['request']}", 8.6, False, 14)
    y += 8
    line("CATATAN BATAS", 10.5, True, 19)
    excluded_labels = {
        "hidden component identity": "identitas komponen tersembunyi",
        "physical unit count from image count": "jumlah unit fisik dari jumlah foto",
        "material composition or value": "komposisi material atau nilai",
        "toxicity, safety, or disassembly instruction": "toksisitas, keselamatan, atau instruksi pembongkaran",
    }
    for excluded in state.get("claims_excluded", []):
        line(f"• Tidak menilai {excluded_labels.get(excluded, excluded)}.", 8.4, False, 13)
    payload = document.tobytes(garbage=4, deflate=True)
    document.close()
    return payload
