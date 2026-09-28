"""Render original-photo evidence for every multiview retrieval audit query."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from render_mask_correspondence_audit_20260928 import (
    TEAL,
    decode_mask,
    digest,
    fit,
    font,
    masked_crop,
    original_with_mask,
)


def load_proposals(path: Path, wanted: set[str]) -> dict[str, dict]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            region_id = row["region_id"]
            if region_id in wanted:
                result[region_id] = row
                if len(result) == len(wanted):
                    break
    missing = wanted - set(result)
    if missing:
        raise ValueError(f"Missing proposal rows: {sorted(missing)[:3]}")
    return result


def source_rows(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in rows:
        result.setdefault(str(row["exact_duplicate_canonical_id"]), row)
    return result


def evidence_panel(original: Image.Image, proposal: dict, size=(500, 330)) -> Image.Image:
    mask = decode_mask(proposal)
    mask_processing = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_native = mask_processing.resize(original.size, Image.Resampling.NEAREST)
    overlay = original_with_mask(original, mask_native)
    box = [float(value) for value in proposal["mask_bbox_xyxy_native"]]
    crop = masked_crop(original, mask_native, box)
    full, _ = fit(overlay, (330, size[1]))
    cut, _ = fit(crop, (size[0] - 340, size[1]))
    panel = Image.new("RGB", size, (246, 247, 244))
    panel.paste(full, (0, 0))
    panel.paste(cut, (340, 0))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="dino_foreground_nn_all_r2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions.arm.eq(args.arm)].reset_index(drop=True)
    if len(predictions) != 42:
        raise ValueError("Expected all 42 audit queries for one arm")
    wanted = set(predictions.query_region_id) | set(predictions.candidate_region_id)
    proposals = load_proposals(args.proposals, wanted)
    sources = source_rows(args.image_manifest)

    sheet_paths = []
    rendered_rows = []
    per_sheet = 6
    width, header, row_height = 1500, 105, 455
    for start in range(0, len(predictions), per_sheet):
        subset = predictions.iloc[start:start + per_sheet]
        canvas = Image.new("RGB", (width, header + row_height * len(subset)), (255, 255, 253))
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 18), "Audit pasangan retrieval r2 lengkap", font=font(28, True), fill=(18, 38, 45))
        draw.text((28, 60), "Kiri: kueri; kanan: kandidat teratas. Overlay hijau = mask tersimpan; crop mempertahankan piksel mask.",
                  font=font(16), fill=(55, 72, 78))
        for local_index, row in enumerate(subset.itertuples()):
            y = header + local_index * row_height
            draw.line((20, y, width - 20, y), fill=(205, 211, 208), width=1)
            query_source = sources[str(row.query_parent_id)]
            candidate_source = sources[str(row.candidate_parent_id)]
            query_path = args.raw_root / query_source["source_relative_path"]
            candidate_path = args.raw_root / candidate_source["source_relative_path"]
            for path, source in [(query_path, query_source), (candidate_path, candidate_source)]:
                if not path.is_file() or digest(path) != source["source_sha256"]:
                    raise ValueError(f"Source verification failed: {path}")
            with Image.open(query_path) as opened:
                query_image = ImageOps.exif_transpose(opened).convert("RGB")
            with Image.open(candidate_path) as opened:
                candidate_image = ImageOps.exif_transpose(opened).convert("RGB")
            query_panel = evidence_panel(query_image, proposals[row.query_region_id])
            candidate_panel = evidence_panel(candidate_image, proposals[row.candidate_region_id])
            canvas.paste(query_panel, (28, y + 65))
            canvas.paste(candidate_panel, (772, y + 65))
            same_family = int(row.query_family_internal) == int(row.candidate_family_internal)
            visible = bool(row.gold_visible_family)
            status = "keluarga sama" if same_family else "keluarga berbeda"
            status_color = TEAL if same_family else (176, 64, 49)
            draw.text((28, y + 15), f"{start + local_index + 1:02d}. {row.query_visual_name_id}",
                      font=font(19, True), fill=(18, 38, 45))
            draw.text((565, y + 16), f"audit terlihat: {'ya' if visible else 'tidak'}", font=font(15), fill=(55, 72, 78))
            draw.text((772, y + 15), f"{status}: F{int(row.query_family_internal):04d} → F{int(row.candidate_family_internal):04d}",
                      font=font(17, True), fill=status_color)
            draw.text((772, y + 42),
                      f"DINO fg {row.dino_foreground_cosine:.3f}  |  box {row.dino_box_cosine:.3f}  |  SigLIP crop {row.siglip2_crop_cosine:.3f}",
                      font=font(14), fill=(55, 72, 78))
            draw.text((28, y + 405), f"kueri: {query_source['source_relative_path']}", font=font(13), fill=(68, 78, 82))
            draw.text((772, y + 405), f"kandidat: {candidate_source['source_relative_path']}", font=font(13), fill=(68, 78, 82))
            rendered_rows.append({
                "audit_index": start + local_index + 1,
                "query_region_id": row.query_region_id,
                "candidate_region_id": row.candidate_region_id,
                "query_source_relative_path": query_source["source_relative_path"],
                "candidate_source_relative_path": candidate_source["source_relative_path"],
                "query_source_sha256": query_source["source_sha256"],
                "candidate_source_sha256": candidate_source["source_sha256"],
                "gold_visible_family": visible,
                "same_r2_family": same_family,
                "pair_semantic_correspondence": "pending_visual_adjudication",
                "pair_evidence_note": "pending_visual_adjudication",
            })
        sheet = args.output / f"retrieval_pair_audit_{start // per_sheet + 1:02d}.png"
        canvas.save(sheet, dpi=(150, 150))
        sheet_paths.append(sheet)

    manifest = args.output / "pair_audit_manifest.csv"
    pd.DataFrame(rendered_rows).to_csv(manifest, index=False)
    summary = {
        "status": "rendered_original_photo_pair_audit",
        "arm": args.arm,
        "pairs": len(rendered_rows),
        "sheets": len(sheet_paths),
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in sheet_paths},
        "manifest_sha256": digest(manifest),
        "claim_limit": "Rendering and r2-family agreement do not adjudicate semantic pair correspondence.",
    }
    (args.output / "render_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
