"""Render top-five family candidates for visible queries missed at rank one."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from render_mask_correspondence_audit_20260928 import decode_mask, digest, fit, font, masked_crop, original_with_mask


def load_proposals(path: Path, wanted: set[str]) -> dict[str, dict]:
    output = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["region_id"] in wanted:
                output[row["region_id"]] = row
                if len(output) == len(wanted):
                    break
    if set(output) != wanted:
        raise ValueError("Missing proposal rows")
    return output


def panel(image: Image.Image, proposal: dict, size=(250, 250)) -> Image.Image:
    mask = decode_mask(proposal)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(image.size, Image.Resampling.NEAREST)
    overlay = original_with_mask(image, mask_image)
    box = [float(value) for value in proposal["mask_bbox_xyxy_native"]]
    crop = masked_crop(image, mask_image, box)
    full, _ = fit(overlay, (size[0], 155))
    cut, _ = fit(crop, (size[0], size[1] - 160))
    result = Image.new("RGB", size, (247, 248, 245))
    result.paste(full, (0, 0))
    result.paste(cut, (0, 160))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top5", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    top5 = pd.read_csv(args.top5)
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions.arm.eq("dino_foreground_nn_all_r2")]
    missed = predictions[predictions.gold_visible_family & ~predictions.correct_visible_family]
    if len(missed) != 3:
        raise ValueError("Expected three visible rank-one misses")
    query_ids = missed.query_region_id.tolist()
    top5 = top5[top5.query_region_id.isin(query_ids)].copy()
    wanted = set(top5.candidate_region_id) | set(query_ids)
    proposals = load_proposals(args.proposals, wanted)
    manifest_rows = json.loads(args.image_manifest.read_text(encoding="utf-8"))
    sources = {}
    for row in manifest_rows:
        sources.setdefault(str(row["exact_duplicate_canonical_id"]), row)

    width, header, row_height = 1780, 100, 380
    canvas = Image.new("RGB", (width, header + row_height * len(query_ids)), (255, 255, 253))
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 16), "Pemulihan keluarga pada lima kandidat teratas", font=font(27, True), fill=(18, 38, 45))
    draw.text((25, 57), "Tiga kueri terlihat yang terlewat pada rank 1; hijau menandai keluarga r2 yang diharapkan.",
              font=font(16), fill=(55, 72, 78))
    output_rows = []
    for row_index, query_id in enumerate(query_ids):
        y = header + row_index * row_height
        draw.line((18, y, width - 18, y), fill=(205, 211, 208), width=1)
        query_rows = top5[top5.query_region_id.eq(query_id)].sort_values("family_rank")
        query = query_rows.iloc[0]
        qsource = sources[str(query.query_parent_id)]
        qpath = args.raw_root / qsource["source_relative_path"]
        if not qpath.is_file() or digest(qpath) != qsource["source_sha256"]:
            raise ValueError(f"Source verification failed: {qpath}")
        with Image.open(qpath) as opened:
            qimage = ImageOps.exif_transpose(opened).convert("RGB")
        canvas.paste(panel(qimage, proposals[query_id], (260, 270)), (25, y + 67))
        draw.text((25, y + 17), str(query.query_visual_name_id), font=font(18, True), fill=(18, 38, 45))
        draw.text((25, y + 43), f"kueri F{int(query.query_family_internal):04d}", font=font(14), fill=(55, 72, 78))
        for column, candidate in enumerate(query_rows.itertuples()):
            source = sources[str(candidate.candidate_parent_id)]
            path = args.raw_root / source["source_relative_path"]
            if not path.is_file() or digest(path) != source["source_sha256"]:
                raise ValueError(f"Source verification failed: {path}")
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            x = 315 + column * 290
            canvas.paste(panel(image, proposals[candidate.candidate_region_id], (260, 270)), (x, y + 67))
            color = (0, 136, 145) if bool(candidate.is_expected_family) else (87, 96, 98)
            draw.text((x, y + 17), f"rank {int(candidate.family_rank)} · F{int(candidate.candidate_family_internal):04d}",
                      font=font(16, True), fill=color)
            draw.text((x, y + 43), f"DINO {candidate.dino_foreground_cosine:.3f}", font=font(13), fill=(55, 72, 78))
            output_rows.append({
                "query_region_id": query_id,
                "family_rank": int(candidate.family_rank),
                "candidate_region_id": candidate.candidate_region_id,
                "candidate_family_internal": int(candidate.candidate_family_internal),
                "is_expected_family": bool(candidate.is_expected_family),
                "candidate_source_relative_path": source["source_relative_path"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(180, 180))
    manifest = args.output.with_suffix(".csv")
    pd.DataFrame(output_rows).to_csv(manifest, index=False)
    summary = {
        "status": "rendered_top5_recovery_audit",
        "queries": len(query_ids),
        "candidate_rows": len(output_rows),
        "image_sha256": digest(args.output),
        "manifest_sha256": digest(manifest),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
