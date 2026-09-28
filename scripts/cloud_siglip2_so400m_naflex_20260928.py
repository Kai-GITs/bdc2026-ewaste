"""Extract full canonical BDC image features with official SigLIP2 NaFlex processing.

Uses a compatible vision runtime, persistent cache and private artifact storage.
No labels, prompts, or old SigLIP transform enter this bank.
"""

import csv
import hashlib
import io
import json
import os
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["HF_HOME"] = "/research-cache/hf"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor, __version__ as transformers_version


MODEL = "google/siglip2-so400m-patch16-naflex"
MODEL_REV = "cc24074f717b612951c2dead130904ab9b65a81e"
WEIGHT_SHA = "11a61a2068800d5f4f35cb041c1fea25de86ce87725e4c97a9ed046d0b22c076"
SOURCE_LOCAL = os.environ.get("BDC_SOURCE_ZIP")
SOURCE = os.environ.get("BDC_SOURCE_DATASET_REPO")
SOURCE_FILE = os.environ.get("BDC_SOURCE_DATASET_FILE", "bdc-official-2026.zip")
SOURCE_REV = os.environ.get("BDC_SOURCE_DATASET_REVISION")
SOURCE_MANIFEST_SHA = "697f8ef83052c860782263c7218b0a9bdf7a1611586734b21ed7846f032e5a81"
REPO = "Fin000/codex-research-workspace"
ROI_PATH = "bdc2026-ewaste/track-a/runs/20260924-054638-189f34ad/roi_index.csv"
ROI_SHA = "d4d0b6f67ce9821f38fa3f02ccd7251f09b71a17059882257ab6d720408ff0df"
PREFIX = "bdc2026-ewaste/final-study-20260928/siglip2-so400m-naflex-r1"
OUT = Path("/research-cache/bdc2026/final-study-20260928/siglip2-so400m-naflex-r1")
OUT.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def electronic_relative(name):
    normalized = name.replace("\\", "/")
    marker = "train/1_Electronic/"
    position = normalized.find(marker)
    return None if position < 0 else normalized[position + len("train/"):]


token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
assert api.repo_info(REPO, repo_type="dataset").private
if SOURCE_LOCAL:
    source_path = Path(SOURCE_LOCAL)
else:
    if not SOURCE:
        raise RuntimeError("Set BDC_SOURCE_ZIP or BDC_SOURCE_DATASET_REPO.")
    source_path = Path(hf_hub_download(
        SOURCE, SOURCE_FILE, repo_type="dataset", revision=SOURCE_REV,
        token=os.environ.get("BDC_SOURCE_DATASET_TOKEN") or False))
assert source_path.stat().st_size == 1368394793
manifest_path = Path("/research-cache/bdc2026/source-verify/electronic_image_hashes.csv")
if not manifest_path.exists():
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_path) as zf:
        members = [(m, electronic_relative(m.filename)) for m in zf.infolist()
                   if not m.is_dir()]
        members = [(m, relative) for m, relative in members if relative is not None]
        assert len(members) == 3961
        rows = []
        for member, relative in members:
            h = hashlib.sha256()
            with zf.open(member) as reader:
                for block in iter(lambda: reader.read(1 << 20), b""):
                    h.update(block)
            rows.append((relative, member.file_size, h.hexdigest(), member.filename))
        rows.sort(key=lambda x: x[0])
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("path", "bytes", "sha256", "zip_member"))
        writer.writerows(rows)
assert digest(manifest_path) == SOURCE_MANIFEST_SHA
with manifest_path.open(encoding="utf-8", newline="") as f:
    manifest = list(csv.DictReader(f))
assert len(manifest) == 3961
source_by_sha = {}
for row in manifest:
    source_by_sha.setdefault(row["sha256"], row)
assert len(source_by_sha) == 3931

roi_path = Path(hf_hub_download(REPO, ROI_PATH, repo_type="dataset", token=token))
assert digest(roi_path) == ROI_SHA
sha_to_id = {}
with roi_path.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        old = sha_to_id.setdefault(row["source_sha256"], row["canonical_id"])
        assert old == row["canonical_id"]
assert set(sha_to_id) == set(source_by_sha)
items = [(sha_to_id[sha], sha, source_by_sha[sha]["zip_member"])
         for sha in sorted(source_by_sha)]
assert len(items) == 3931
index_path = OUT / "index.csv"
if not index_path.exists():
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("row", "canonical_id", "sha256", "zip_member"))
        writer.writerows((i, *item) for i, item in enumerate(items))
INDEX_SHA = digest(index_path)

weight_path = Path(hf_hub_download(MODEL, "model.safetensors", revision=MODEL_REV, token=False))
assert digest(weight_path) == WEIGHT_SHA
processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=False)
model = AutoModel.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=False,
                                  use_safetensors=True, dtype=torch.float16).eval().cuda()
torch.set_num_threads(4)
assert "Siglip2ImageProcessor" in type(processor.image_processor).__name__
assert processor.image_processor.max_num_patches == 256
assert processor.image_processor.patch_size == 16
print(json.dumps({"event": "inputs_ready", "canonical": len(items), "model": MODEL,
                  "model_revision": MODEL_REV, "model_sha256": WEIGHT_SHA,
                  "processor": type(processor.image_processor).__name__,
                  "transformers": transformers_version, "torch": torch.__version__}), flush=True)


def load_one(item):
    ident, sha, member = item
    with zipfile.ZipFile(source_path) as zf:
        raw = zf.read(member)
    assert hashlib.sha256(raw).hexdigest() == sha
    with Image.open(io.BytesIO(raw)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


feature_path = OUT / "features.npy"
progress_path = OUT / "progress.json"
if progress_path.exists():
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["index_sha256"] == INDEX_SHA and progress["model_revision"] == MODEL_REV
    start_at = int(progress["completed_images"])
    features = np.load(feature_path, mmap_mode="r+")
    assert features.shape[0] == 3931 and features.dtype == np.float16
else:
    assert not feature_path.exists()
    start_at = 0
    features = None
assert 0 <= start_at <= len(items)
began = time.monotonic()
batch_size = 8
shapes = Counter()
with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
    for begin in range(start_at, len(items), batch_size):
        end = min(begin + batch_size, len(items))
        photos = list(pool.map(load_one, items[begin:end]))
        batch = processor(images=photos, max_num_patches=256, return_tensors="pt")
        assert {"pixel_values", "pixel_attention_mask", "spatial_shapes"}.issubset(batch.keys()), list(batch)
        assert len(batch["pixel_values"]) == end - begin
        for shape in batch["spatial_shapes"].tolist():
            shapes[tuple(shape)] += 1
        batch = {key: val.cuda(non_blocking=True) for key, val in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            result = model.get_image_features(**batch)
        vectors = result.pooler_output if hasattr(result, "pooler_output") else result
        vectors = vectors.float().cpu().numpy()
        assert vectors.ndim == 2 and vectors.shape[0] == end - begin and np.isfinite(vectors).all()
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        if features is None:
            features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16,
                                                 shape=(3931, vectors.shape[1]))
        assert vectors.shape[1] == features.shape[1]
        features[begin:end] = vectors.astype(np.float16)
        if end % 64 < batch_size or end == len(items):
            features.flush()
            write_json(progress_path, {"completed_images": end, "canonical_total": len(items),
                                       "index_sha256": INDEX_SHA, "model_revision": MODEL_REV})
            print(json.dumps({"event": "embedded", "count": end,
                              "elapsed_seconds": round(time.monotonic() - began, 2)}), flush=True)

assert features is not None and np.isfinite(features).all()
assert np.linalg.norm(np.asarray(features, dtype=np.float32), axis=1).min() > .99
metrics = {"status": "complete", "canonical_images": 3931, "parent_records": 3961,
           "feature_shape": list(features.shape), "model": MODEL, "model_revision": MODEL_REV,
           "model_weight_sha256": WEIGHT_SHA, "source_revision": SOURCE_REV,
           "source_manifest_sha256": SOURCE_MANIFEST_SHA, "roi_index_sha256": ROI_SHA,
           "index_sha256": INDEX_SHA, "processor": type(processor.image_processor).__name__,
           "preprocess": "official AutoProcessor NaFlex; EXIF oriented RGB; max_num_patches=256; no legacy SigLIP letterbox",
           "pooling": "AutoModel.get_image_features(...).pooler_output; L2 normalization",
           "spatial_shape_counts_this_attempt": {str(k): v for k, v in shapes.items()},
           "torch": torch.__version__, "transformers": transformers_version,
           "elapsed_seconds_this_attempt": round(time.monotonic() - began, 2),
           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())}
write_json(OUT / "metrics.json", metrics)
names = ["features.npy", "index.csv", "metrics.json"]
checks = {name: {"bytes": (OUT/name).stat().st_size, "sha256": digest(OUT/name)} for name in names}
commit = api.upload_folder(folder_path=str(OUT), path_in_repo=PREFIX,
                           repo_id=REPO, repo_type="dataset", allow_patterns=names,
                           commit_message="Private full BDC SigLIP2 So400m NaFlex embeddings")
for name in names:
    remote = Path(hf_hub_download(REPO, PREFIX+"/"+name, repo_type="dataset",
                                  revision=commit.oid, token=token))
    assert remote.stat().st_size == checks[name]["bytes"] and digest(remote) == checks[name]["sha256"]
print(json.dumps({"event": "complete_private_verified", "revision": commit.oid,
                  "prefix": PREFIX, "metrics": metrics, "files": checks}), flush=True)
