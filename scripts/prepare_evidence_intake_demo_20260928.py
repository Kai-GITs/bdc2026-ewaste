"""Prepare a reproducible intake demo from frozen r3 supports and real BDC photos."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


EXAMPLES = [
    ("93bcd9f51f027f41", "3c924a774442d2f716fe2dda"),
    ("297b5776844a8d24", "32e33062d553da6a63546e13"),
    ("f14e54d29cd238a9", "232533266c1b0e21ddf8e282"),
    ("e31f2b952c857db8", "1024fc06f5bc23b4bbba0cad"),
    ("ee677fbd3c36d3fd", "f87fe2e4aa3245866fb0900e"),
    ("96eefdc0ef05bd36", "87c7f60dd2cbe58cf900b885"),
]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--supports", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(args.assignments).set_index("canonical_id")
    supports = pd.read_csv(args.supports).set_index("support_region_id", drop=False)
    wanted = {region for _, region in EXAMPLES}
    proposals = {}
    with gzip.open(args.proposals, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["region_id"] in wanted:
                proposals[row["region_id"]] = row
    if set(proposals) != wanted:
        raise ValueError("Demo proposal set is incomplete")

    photos, family_support, ownership = [], [], []
    for parent, region in EXAMPLES:
        source = assignments.loc[parent]
        path = args.raw_root / source.source_relative_path
        if file_sha(path) != source.source_sha256:
            raise ValueError(f"Source SHA mismatch: {path}")
        support = supports.loc[region]
        if isinstance(support, pd.DataFrame):
            support = support.iloc[0]
        proposal = proposals[region]
        display_ambiguity = str(support.descriptive_family_key) == "dark_display_surface"
        photos.append({
            "photo_id": parent,
            "sha256": source.source_sha256,
            "image_path": source.source_relative_path.replace("\\", "/"),
            "width": int(source.width),
            "height": int(source.height),
            "readable": True,
            "closed_casing": display_ambiguity,
            "sensor_ambiguity_cue": display_ambiguity,
            "ambiguity_kind": "display_state" if display_ambiguity else None,
        })
        family_support.append({
            "photo_id": parent,
            "family_key": support.descriptive_family_key,
            "family_name": support.descriptive_name_id,
            "machine_family_id": f"F{int(support.source_family_internal):04d}",
            "region_id": region,
            "bbox_xyxy": proposal["mask_bbox_xyxy_native"],
            "mask_sha256": proposal["mask_sha256"],
            "support_strength": float(support.two_view_consensus_strength),
            "support_status": support.support_status,
            "claim_gate_status": "r3_core_requires_human_confirmation",
        })
        ownership.append({
            "photo_id": parent,
            "region_id": region,
            "ownership_status": "geometric_parent_candidate",
            "owner_region_id": parent,
        })

    # Demonstrate exact-byte intake consolidation with a second receipt of one
    # real image; this is a workflow fixture, not a new dataset observation.
    duplicate = dict(photos[0])
    duplicate["photo_id"] = photos[0]["photo_id"] + "-receipt-copy"
    photos.insert(1, duplicate)

    batch = {
        "dossier_id": "dossier_bdc_demo_20260928",
        "batch_id": "BDC-DEMO-20260928",
        "discovery_revision": "region-family-census-r2+consensus-r3",
        "family_name_map_revision": "family-adjudication-r3-20260928",
        "photos": photos,
        "family_support": family_support,
        "ownership_evidence": ownership,
        "demo_scope": "six real r3 core supports plus one repeated file receipt",
    }
    batch_path = args.output / "frozen_batch.json"
    batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {
        "status": "complete",
        "batch_sha256": file_sha(batch_path),
        "received_records": len(photos),
        "unique_exact_images_expected": 6,
        "supports": len(family_support),
        "original_photo_sha_verified": True,
        "claim_limit": "Workflow demo; repeated receipt is intentionally duplicated and is not a seventh observation.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
