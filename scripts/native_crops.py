"""Export deterministic, non-resampled central crops for actual image inspection."""

import argparse
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

from intake import io_path, save_json
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    if not 32 <= args.size <= 2048:
        parser.error("Crop size must be in 32..2048.")
    root = io_path(args.source_root).resolve(strict=True)
    if io_path(args.output).resolve().is_relative_to(root):
        parser.error("Output must be outside the source tree.")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records) or any(not re.fullmatch(r"[0-9a-f]{16}", key) for key in by_id):
        parser.error("Manifest IDs must be unique 16-character lowercase hexadecimal strings.")
    if len(set(args.ids)) != len(args.ids) or not set(args.ids) <= by_id.keys():
        parser.error("IDs must be distinct and present in the manifest.")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = []
    for sample_id in args.ids:
        row = by_id[sample_id]
        source = (root / row["path"]).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("Source path escapes the dataset root.")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("Source bytes differ from the intake snapshot.")
        with Image.open(BytesIO(payload)) as original:
            oriented = ImageOps.exif_transpose(original)
            width, height = oriented.size
            left, top = max(0, (width - args.size) // 2), max(0, (height - args.size) // 2)
            box = (left, top, min(width, left + args.size), min(height, top + args.size))
            oriented.crop(box).save(args.output / f"{sample_id}.png")
        receipt.append(
            {
                "id": sample_id,
                "path": row["path"],
                "source_sha256": row["sha256"],
                "oriented_size": [width, height],
                "crop_xyxy": box,
                "resampled": False,
                "status": "PENDING_VISUAL_INSPECTION",
            }
        )
    save_json(args.output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
