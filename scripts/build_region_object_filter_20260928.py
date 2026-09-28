"""Apply a class-free geometric coherence filter to the completed SAM3 bank.

The filter responds to a measured failure mode in the first region graph:
families dominated by image corners, prompt-box boundaries, solid rectangles,
and global masks.  It uses mask geometry only and never sees family assignments,
product labels, filenames, or target component concepts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


MIN_AREA_FRACTION = 0.005
MAX_AREA_FRACTION = 0.65
MIN_BBOX_SIDE_NATIVE = 24.0
MIN_MASK_OCCUPANCY = 0.12
MAX_MASK_OCCUPANCY = 0.92
MAX_IMAGE_BOUNDARY_EDGES = 0
MAX_PROMPT_BOUNDARY_EDGES = 1
BOUNDARY_TOLERANCE_PROCESSING_PX = 1.5


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    decisions = []
    with gzip.open(args.proposals, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["selection_status"] != "selected":
                continue
            bbox = [float(value) for value in row["mask_bbox_xyxy_processing"]]
            prompt = [float(value) for value in row["prompt_box_xyxy_processing"]]
            width, height = float(row["processing_width"]), float(row["processing_height"])
            bbox_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            occupancy = float(row["area_pixels"]) / bbox_area
            image_touch = sum((
                bbox[0] <= BOUNDARY_TOLERANCE_PROCESSING_PX,
                bbox[1] <= BOUNDARY_TOLERANCE_PROCESSING_PX,
                bbox[2] >= width - BOUNDARY_TOLERANCE_PROCESSING_PX,
                bbox[3] >= height - BOUNDARY_TOLERANCE_PROCESSING_PX,
            ))
            prompt_touch = sum(
                abs(bbox[index] - prompt[index]) <= BOUNDARY_TOLERANCE_PROCESSING_PX
                for index in range(4)
            )
            reasons = []
            if not MIN_AREA_FRACTION <= float(row["area_fraction"]) <= MAX_AREA_FRACTION:
                reasons.append("area_fraction")
            if float(row["bbox_min_side_native"]) < MIN_BBOX_SIDE_NATIVE:
                reasons.append("native_resolution")
            if not MIN_MASK_OCCUPANCY <= occupancy <= MAX_MASK_OCCUPANCY:
                reasons.append("mask_occupancy")
            if image_touch > MAX_IMAGE_BOUNDARY_EDGES:
                reasons.append("image_boundary")
            if prompt_touch > MAX_PROMPT_BOUNDARY_EDGES:
                reasons.append("prompt_boundary")
            decisions.append({
                "feature_row": int(row["feature_row"]),
                "region_id": row["region_id"],
                "parent_id": row["parent_id"],
                "source_sha256": row["source_sha256"],
                "prompt_name": row["prompt_name"],
                "sam_score": row["score"],
                "area_fraction": row["area_fraction"],
                "bbox_min_side_native": row["bbox_min_side_native"],
                "mask_occupancy_in_bbox": occupancy,
                "image_boundary_edges": image_touch,
                "prompt_boundary_edges": prompt_touch,
                "eligible_object_region": int(not reasons),
                "exclusion_reasons": ";".join(reasons),
            })

    assert [row["feature_row"] for row in decisions] == list(range(len(decisions)))
    eligible = [row for row in decisions if row["eligible_object_region"]]
    assert eligible
    fields = list(decisions[0])
    decision_path = args.output / "region_filter_decisions.csv.gz"
    with gzip.open(decision_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decisions)
    eligible_path = args.output / "eligible_regions.csv.gz"
    with gzip.open(eligible_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eligible)

    parents = {row["parent_id"] for row in decisions}
    eligible_parents = {row["parent_id"] for row in eligible}
    reason_counts = Counter(
        reason for row in decisions for reason in row["exclusion_reasons"].split(";") if reason
    )
    metrics = {
        "status": "complete_class_free_region_object_filter",
        "input_selected_regions": len(decisions),
        "eligible_regions": len(eligible),
        "excluded_regions": len(decisions) - len(eligible),
        "input_parents": len(parents),
        "parents_with_eligible_regions": len(eligible_parents),
        "parents_retained_fraction": len(eligible_parents) / len(parents),
        "maximum_eligible_regions_per_parent": max(Counter(
            row["parent_id"] for row in eligible).values()),
        "exclusion_reason_counts_nonexclusive": dict(sorted(reason_counts.items())),
        "criteria": {
            "area_fraction": [MIN_AREA_FRACTION, MAX_AREA_FRACTION],
            "minimum_bbox_side_native": MIN_BBOX_SIDE_NATIVE,
            "mask_occupancy_in_bbox": [MIN_MASK_OCCUPANCY, MAX_MASK_OCCUPANCY],
            "maximum_image_boundary_edges": MAX_IMAGE_BOUNDARY_EDGES,
            "maximum_prompt_boundary_edges": MAX_PROMPT_BOUNDARY_EDGES,
            "boundary_tolerance_processing_pixels": BOUNDARY_TOLERANCE_PROCESSING_PX,
        },
        "rationale": "Remove the observed corner, frame, prompt-cut, nearly empty, solid-box and global-mask mechanisms before reclustering.",
        "claim_limit": "Eligibility is geometric quality control, not component identity or semantic validity. Photos without eligible regions remain represented in the global photo graph.",
        "input": {"path": str(args.proposals), "sha256": digest(args.proposals)},
        "outputs": {
            "region_filter_decisions.csv.gz": {
                "sha256": digest(decision_path), "bytes": decision_path.stat().st_size},
            "eligible_regions.csv.gz": {
                "sha256": digest(eligible_path), "bytes": eligible_path.stat().st_size},
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
