"""Verified original-photo and stored-mask rendering primitives."""
import base64, hashlib, struct, zlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def decode_mask(row: dict) -> np.ndarray:
    counts = np.frombuffer(
        zlib.decompress(base64.b64decode(row["mask_rle_zlib_base64"])),
        dtype="<u4",
    )
    flat = np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts.astype(np.int64))
    height, width = map(int, row["mask_shape_hw"])
    if flat.size != height * width:
        raise ValueError("RLE size mismatch")
    mask = flat.reshape(height, width)
    check = hashlib.sha256()
    check.update(struct.pack("<II", height, width))
    check.update(np.packbits(mask, bitorder="little").tobytes())
    if check.hexdigest() != row["mask_sha256"]:
        raise ValueError("Mask hash mismatch")
    return mask

def load_photo(path: Path, expected_sha: str) -> Image.Image:
    if not path.is_file() or sha256(path) != expected_sha:
        raise ValueError(f"Source verification failed: {path}")
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")

def crop_photo_and_mask(photo: Image.Image, mask: np.ndarray, bbox, pad: float = 0.32):
    x0, y0, x1, y1 = map(float, bbox)
    width, height = max(x1 - x0, 1), max(y1 - y0, 1)
    x0 = max(0, int(np.floor(x0 - width * pad)))
    y0 = max(0, int(np.floor(y0 - height * pad)))
    x1 = min(photo.width, int(np.ceil(x1 + width * pad)))
    y1 = min(photo.height, int(np.ceil(y1 + height * pad)))
    return photo.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1], (x0, y0, x1, y1)

def style_image_axis(ax, color: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(0.65)

def draw_whole_evidence(ax, photo: Image.Image, proposal: dict, color: str) -> None:
    mask = decode_mask(proposal)
    if mask.shape != (photo.height, photo.width):
        mask = np.array(
            Image.fromarray(mask * 255).resize(photo.size, Image.Resampling.NEAREST)
        ) > 127
    ax.imshow(photo)
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.0)
    style_image_axis(ax, color)
