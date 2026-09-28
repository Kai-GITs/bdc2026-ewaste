"""Extract the frozen DINOv3-L whole-image bank on a CUDA cloud worker.

The script is provider agnostic: mount or download the pinned BDC zip, image
manifest and DINOv3 safetensors first. It writes a new output directory and
never merges with another model or preprocessing bank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize, ToDtype, ToImage


MODEL = "timm/vit_large_patch16_dinov3.lvd1689m"
MODEL_REVISION = "30c1109559f65dea34316b0d4842d35c5771fe11"
MODEL_SHA256 = "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to mix or overwrite an embedding bank: {args.output}")
    if digest(args.weights) != MODEL_SHA256:
        raise ValueError("DINOv3 weight hash mismatch")

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3961:
        raise ValueError("Expected 3,961 parent records")
    by_sha: dict[str, dict] = {}
    for row in rows:
        by_sha.setdefault(row["sha256"], row)
    items = [(sha, by_sha[sha]) for sha in sorted(by_sha)]
    if len(items) != 3931:
        raise ValueError("Expected 3,931 byte-unique images")

    model = timm.create_model(
        "vit_large_patch16_dinov3",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )
    model.load_state_dict(load_file(str(args.weights)), strict=True)
    model = model.eval().cuda()
    transform = Compose([
        ToImage(),
        Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(256),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    archive = zipfile.ZipFile(args.source_zip)

    def load_one(item: tuple[str, dict]) -> tuple[torch.Tensor, dict]:
        expected_sha, row = item
        raw = archive.read(row["zip_member"])
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError(f"Source hash mismatch: {row['zip_member']}")
        with Image.open(io.BytesIO(raw)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        return transform(rgb), row

    started = time.perf_counter()
    features = np.empty((len(items), 1024), dtype=np.float32)
    index_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for begin in range(0, len(items), args.batch_size):
            loaded = list(pool.map(load_one, items[begin : begin + args.batch_size]))
            batch = torch.stack([item[0] for item in loaded]).cuda(non_blocking=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                encoded = model(batch)
                encoded = torch.nn.functional.normalize(encoded.float(), dim=1)
            features[begin : begin + len(loaded)] = encoded.cpu().numpy()
            for offset, (_, row) in enumerate(loaded):
                index_rows.append({
                    "row": begin + offset,
                    "source_sha256": row["sha256"],
                    "source_relative_path": row["path"],
                })

    args.output.mkdir(parents=True)
    feature_path = args.output / "features.npy"
    index_path = args.output / "index.csv"
    np.save(feature_path, features)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "source_sha256", "source_relative_path"])
        writer.writeheader()
        writer.writerows(index_rows)
    metrics = {
        "status": "complete",
        "model": MODEL,
        "revision": MODEL_REVISION,
        "weights_sha256": MODEL_SHA256,
        "canonical_images": len(items),
        "feature_shape": list(features.shape),
        "preprocessing": "EXIF transpose; RGB; bicubic resize shortest edge 256; center crop 256; ImageNet normalization",
        "pooling": "timm global_pool=avg; L2",
        "source_manifest_sha256": digest(args.manifest),
        "features_sha256": digest(feature_path),
        "index_sha256": digest(index_path),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
