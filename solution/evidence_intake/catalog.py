"""Read and query the frozen collection discovery catalog.

The catalog exposes the precomputed r2 partition and r3 support status.  It
does not assign a family to a new photo and never upgrades model evidence to
an observation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_catalog(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported collection catalog schema")
    if not isinstance(value.get("families"), list) or not isinstance(value.get("items"), list):
        raise ValueError("catalog needs families and items lists")
    region_ids = [row.get("region_id") for row in value["items"]]
    if any(not isinstance(region_id, str) or not region_id for region_id in region_ids):
        raise ValueError("every catalog item needs a region_id")
    if len(region_ids) != len(set(region_ids)):
        raise ValueError("catalog region_id values must be unique")
    return value


def catalog_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    return {
        **catalog["summary"],
        "scope": catalog["scope"],
        "limitations": catalog["limitations"],
    }


def list_families(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        catalog["families"],
        key=lambda row: (-int(row["unique_parents"]), int(row["family_internal"])),
    )


def family_items(
    catalog: dict[str, Any], family_key: str | None = None, limit: int = 24, offset: int = 0,
    family_internal: int | None = None,
) -> dict[str, Any]:
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    if family_internal is not None:
        rows = [row for row in catalog["items"] if row["family_internal"] == int(family_internal)]
    elif family_key:
        rows = [row for row in catalog["items"] if row["family_key"] == family_key]
    else:
        raise ValueError("family_key or family_internal is required")
    rows.sort(key=lambda row: (not row["is_r3_core"], -float(row["family_affinity"]), row["region_id"]))
    return {"total": len(rows), "offset": offset, "limit": limit, "items": rows[offset:offset + limit]}


def query_region(catalog: dict[str, Any], region_id: str) -> dict[str, Any]:
    by_region = {row["region_id"]: row for row in catalog["items"]}
    if region_id not in by_region:
        raise KeyError(f"unknown catalog region: {region_id}")
    item = by_region[region_id]
    neighbors = [
        {**by_region[neighbor["region_id"]], "similarity": neighbor["similarity"]}
        for neighbor in item.get("neighbors", [])
        if neighbor["region_id"] in by_region
    ]
    candidates = [
        {
            **by_region[candidate["region_id"]],
            "similarity": candidate["similarity"],
            "same_family": candidate["same_family"],
            "same_global_context": (
                by_region[candidate["region_id"]].get("global_community")
                == item.get("global_community")
            ),
        }
        for candidate in item.get("candidates", [])
        if candidate["region_id"] in by_region
    ]
    return {"query": item, "neighbors": neighbors, "candidates": candidates}


def photo_items_by_sha(catalog: dict[str, Any], sha256: str) -> dict[str, Any]:
    value = str(sha256).strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    rows = [row for row in catalog["items"] if str(row["source_sha256"]).lower() == value]
    if not rows:
        raise KeyError("uploaded bytes do not match a photo in the frozen collection bank")
    rows.sort(key=lambda row: (-float(row["family_affinity"]), row["region_id"]))
    return {
        "match_scope": "exact bytes already present in the frozen collection",
        "new_photo_inference_performed": False,
        "sha256": value,
        "items": rows,
    }


def build_batch_from_regions(
    catalog: dict[str, Any], region_ids: list[str], batch_id: str
) -> dict[str, Any]:
    if not isinstance(region_ids, list) or not region_ids:
        raise ValueError("region_ids must be a non-empty list")
    by_region = {row["region_id"]: row for row in catalog["items"]}
    unknown = sorted(set(region_ids) - set(by_region))
    if unknown:
        raise KeyError(f"unknown catalog regions: {unknown}")
    selected = [by_region[value] for value in dict.fromkeys(region_ids)]
    photos_by_id: dict[str, dict[str, Any]] = {}
    supports = []
    ownership = []
    for item in selected:
        photos_by_id.setdefault(item["parent_id"], {
            "photo_id": item["parent_id"],
            "sha256": item["source_sha256"],
            "image_path": item["source_relative_path"],
            "width": item["width"],
            "height": item["height"],
            "readable": True,
        })
        supports.append({
            "photo_id": item["parent_id"],
            "family_key": item["family_key"],
            "family_name": item["family_name"],
            "machine_family_id": item["machine_family_id"],
            "region_id": item["region_id"],
            "bbox_xyxy": item["bbox_xyxy"],
            "mask_sha256": item["mask_sha256"],
            "support_strength": item["family_affinity"],
            "support_status": "r3_core" if item["is_r3_core"] else "r2_discovery_candidate",
            "claim_gate_status": (
                "r3_core_requires_review" if item["is_r3_core"]
                else "r2_candidate_requires_stronger_review"
            ),
        })
        ownership.append({
            "photo_id": item["parent_id"],
            "region_id": item["region_id"],
            "ownership_status": "geometric_parent_candidate",
            "owner_region_id": item["parent_id"],
        })
    return {
        "batch_id": batch_id,
        "discovery_revision": catalog["revisions"]["r2_partition"],
        "family_name_map_revision": catalog["revisions"]["r3_adjudication"],
        "photos": list(photos_by_id.values()),
        "family_support": supports,
        "ownership_evidence": ownership,
        "scope": "selected members of the already analyzed BDC collection",
    }
