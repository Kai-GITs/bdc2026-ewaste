"""Build the full-cohort photo graph required by the active BDC contract.

This stage reuses the frozen DINOv3-L and SigLIP 2 So400m NaFlex banks.  It
fits Leiden communities on a rank-neighbour graph, preserves the historical
Louvain assignment as a comparator, and writes a deterministic two-level
force layout plus inspectable review sheets.  Filenames and review labels do
not enter graph fitting.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / ".local/track-a/manifests/bdc_images.json"
DINO = PROJECT / "cache/dinov3_l_full_v1"
SIGLIP = PROJECT / ".local/siglip2-so400m-naflex-20260928"
OLD = PROJECT / "experiments/final_study_20260928/siglip2_so400m_structure"
IMAGE_ROOT = PROJECT / "data/BDC/train"
OUT = PROJECT / "experiments/multiscale_graph_20260928/global"
SEEDS = (11, 23, 37)
K_VALUES = (15, 30, 45)
RESOLUTIONS = (0.5, 0.8, 1.2)
BASE_K = 30
BASE_RESOLUTION = 0.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with windows_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def windows_path(path: Path) -> Path:
    value = str(path.resolve())
    if len(value) >= 248 and not value.startswith("\\\\?\\"):
        return Path("\\\\?\\" + value)
    return path


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_inputs() -> tuple[list[dict], list[dict], list[str], np.ndarray, np.ndarray]:
    all_records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    canonical = [r for r in all_records if r["image_id"] == r["exact_duplicate_canonical_id"]]
    assert len(all_records) == 3961 and len(canonical) == 3931
    ids = [r["image_id"] for r in canonical]
    assert len(set(ids)) == len(ids)
    hash_to_id = {r["source_sha256"]: r["image_id"] for r in canonical}

    def aligned(bank: Path) -> np.ndarray:
        index = csv_rows(bank / "index.csv")
        bank_ids = [hash_to_id[r.get("sha256",r.get("source_sha256"))] for r in index]
        positions = {ident: i for i, ident in enumerate(bank_ids)}
        assert len(positions) == len(ids) and set(positions) == set(ids)
        raw = np.asarray(np.load(bank / "features.npy", mmap_mode="r"), dtype=np.float32)
        matrix = np.ascontiguousarray(raw[[positions[ident] for ident in ids]])
        assert matrix.ndim == 2 and np.isfinite(matrix).all()
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        return matrix

    return all_records, canonical, ids, aligned(DINO), aligned(SIGLIP)


def ranked_neighbours(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=2)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    out_i = np.empty((len(features), k), dtype=np.int32)
    out_d = np.empty((len(features), k), dtype=np.float32)
    for row in range(len(features)):
        keep = indices[row] != row
        out_i[row] = indices[row][keep][:k]
        out_d[row] = distances[row][keep][:k]
    return out_i, out_d


def directed_rank(neighbours: np.ndarray, k: int) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for source, candidates in enumerate(neighbours[:, :k]):
        for rank, target in enumerate(candidates, 1):
            result[(source, int(target))] = (k + 1 - rank) / k
    return result


def undirected_rank(directed: dict[tuple[int, int], float]) -> dict[tuple[int, int], tuple[float, bool]]:
    result: dict[tuple[int, int], tuple[float, bool]] = {}
    for (source, target), value in directed.items():
        a, b = sorted((source, target))
        reverse = directed.get((target, source), 0.0)
        affinity = max(value, reverse) * (0.35 + 0.65 * min(value, reverse))
        old = result.get((a, b))
        if old is None or affinity > old[0]:
            result[(a, b)] = (affinity, reverse > 0)
    return result


def fused_edges(dino: dict, siglip: dict) -> dict[tuple[int, int], dict[str, float | bool]]:
    result: dict[tuple[int, int], dict[str, float | bool]] = {}
    for edge in set(dino) | set(siglip):
        d, dm = dino.get(edge, (0.0, False))
        s, sm = siglip.get(edge, (0.0, False))
        result[edge] = {
            "dino": d,
            "siglip": s,
            "fused": 0.5 * (d + s) + 0.5 * min(d, s),
            "mutual_dino": dm,
            "mutual_siglip": sm,
        }
    return result


def leiden_partition(edges: dict, n: int, resolution: float, seed: int, weight_key: str) -> np.ndarray:
    pairs = list(edges)
    graph = ig.Graph(n=n, edges=pairs, directed=False)
    weights = [float(edges[p][weight_key] if isinstance(edges[p], dict) else edges[p][0]) for p in pairs]
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=-1,
    )
    raw = np.asarray(partition.membership, dtype=np.int32)
    groups = sorted(np.unique(raw), key=lambda c: (-int(np.sum(raw == c)), int(np.min(np.flatnonzero(raw == c)))))
    remap = {old: new for new, old in enumerate(groups)}
    return np.asarray([remap[int(x)] for x in raw], dtype=np.int32)


def connected_communities(edges: dict, labels: np.ndarray, n: int) -> tuple[int, list[int]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    disconnected = []
    for label in sorted(set(labels.tolist())):
        nodes = np.flatnonzero(labels == label).tolist()
        count = nx.number_connected_components(graph.subgraph(nodes))
        if count != 1:
            disconnected.append(label)
    return len(disconnected), disconnected


def make_layout(edges: dict, labels: np.ndarray, n: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from((a, b, float(meta["fused"])) for (a, b), meta in edges.items())
    quotient = nx.Graph()
    quotient.add_nodes_from(sorted(set(labels.tolist())))
    aggregate: dict[tuple[int, int], float] = defaultdict(float)
    for (a, b), meta in edges.items():
        ca, cb = int(labels[a]), int(labels[b])
        if ca != cb:
            aggregate[tuple(sorted((ca, cb)))] += float(meta["fused"])
    quotient.add_weighted_edges_from((a, b, math.log1p(w)) for (a, b), w in aggregate.items())
    qpos = nx.spring_layout(quotient, seed=280928, weight="weight", iterations=500, scale=8.0)
    counts = Counter(labels.tolist())
    largest = max(counts.values())
    coords = np.zeros((n, 2), dtype=np.float32)
    for label in sorted(counts):
        nodes = np.flatnonzero(labels == label).tolist()
        sub = graph.subgraph(nodes).copy()
        radius = 0.55 + 1.25 * math.sqrt(len(nodes) / largest)
        if len(nodes) == 1:
            local = {nodes[0]: np.zeros(2)}
        else:
            local = nx.spring_layout(sub, seed=280928 + label, weight="weight", iterations=120, scale=radius)
        center = np.asarray(qpos[label], dtype=np.float32)
        for node in nodes:
            coords[node] = center + np.asarray(local[node], dtype=np.float32)

    backbone: list[tuple[int, int]] = []
    for component in nx.connected_components(graph):
        tree = nx.maximum_spanning_tree(graph.subgraph(component), weight="weight")
        backbone.extend((min(a, b), max(a, b)) for a, b in tree.edges())
    backbone_set = set(backbone)
    extras: list[tuple[int, int]] = []
    per_node: Counter[int] = Counter()
    candidates = sorted(
        ((float(meta["fused"]), a, b) for (a, b), meta in edges.items()
         if (meta["mutual_dino"] or meta["mutual_siglip"]) and (a, b) not in backbone_set),
        reverse=True,
    )
    for _, a, b in candidates:
        if per_node[a] >= 2 or per_node[b] >= 2:
            continue
        extras.append((a, b))
        per_node[a] += 1
        per_node[b] += 1
    return coords, sorted(backbone), sorted(extras)


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size)
    except OSError:
        from matplotlib.font_manager import findfont
        return ImageFont.truetype(findfont('DejaVu Sans'), size)


def open_photo(record: dict, cell_size=(230, 182)) -> Image.Image:
    path = IMAGE_ROOT / record["source_relative_path"]
    assert sha256(path) == record["source_sha256"]
    image = ImageOps.exif_transpose(Image.open(windows_path(path))).convert("RGB")
    image.thumbnail((cell_size[0] - 8, cell_size[1] - 26), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", cell_size, "#F7F8F5")
    cell.paste(image, ((cell.width - image.width) // 2, (cell.height - 26 - image.height) // 2))
    return cell


def review_sheets(records: list[dict], features: np.ndarray, labels: np.ndarray) -> list[dict]:
    review_dir = OUT / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(280928)
    groups = sorted(set(labels.tolist()))
    pages = []
    for page_index in range(math.ceil(len(groups) / 3)):
        canvas = Image.new("RGB", (4 * 230, 3 * 220), "#EEF1ED")
        draw = ImageDraw.Draw(canvas)
        page_rows = []
        for block, group in enumerate(groups[page_index * 3:(page_index + 1) * 3]):
            indices = np.flatnonzero(labels == group)
            center = features[indices].mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            similarity = features[indices] @ center
            core = indices[np.argsort(-similarity, kind="stable")[:4]]
            boundary = indices[np.argsort(similarity, kind="stable")[:3]]
            remaining = np.setdiff1d(indices, np.r_[core, boundary])
            random = rng.choice(remaining, size=min(4, len(remaining)), replace=False) if len(remaining) else np.array([], dtype=int)
            chosen = list(dict.fromkeys([*core, *boundary, *random]))[:8]
            y0 = block * 220
            draw.rectangle((0, y0, canvas.width, y0 + 35), fill="#17363B")
            draw.text((10, y0 + 5), f"Komunitas internal {group:02d} | n={len(indices)} | pusat, batas, acak", fill="white", font=font(20, True))
            for slot, row in enumerate(chosen):
                x = (slot % 4) * 230
                y = y0 + 36 + (slot // 4) * 92
                tile = open_photo(records[int(row)], (230, 90))
                canvas.paste(tile, (x, y))
                role = "pusat" if slot < 4 else "batas" if slot < 7 else "acak"
                draw.text((x + 5, y + 68), f"{role} | baris {int(row)}", fill="#17363B", font=font(13, True))
                page_rows.append({"community": group, "row": int(row), "role": role,
                                  "canonical_id": records[int(row)]["image_id"],
                                  "source_sha256": records[int(row)]["source_sha256"]})
        filename = f"leiden_review_{page_index + 1:02d}.jpg"
        canvas.save(review_dir / filename, quality=92)
        pages.append({"path": f"review/{filename}", "rows": page_rows})
    return pages


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global OUT, DINO, SIGLIP, MANIFEST, IMAGE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--dino-dir", type=Path, default=DINO)
    parser.add_argument("--siglip-dir", type=Path, default=SIGLIP)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    DINO, SIGLIP, MANIFEST, IMAGE_ROOT = args.dino_dir, args.siglip_dir, args.manifest, args.image_root
    OUT.mkdir(parents=True, exist_ok=True)
    parents, canonical, ids, dino, siglip = load_inputs()
    n = len(ids)
    max_k = max(K_VALUES)
    dino_nn, _ = ranked_neighbours(dino, max_k)
    sig_nn, _ = ranked_neighbours(siglip, max_k)
    results = {}
    edge_cache = {}
    for k in K_VALUES:
        d = undirected_rank(directed_rank(dino_nn, k))
        s = undirected_rank(directed_rank(sig_nn, k))
        fused = fused_edges(d, s)
        edge_cache[k] = fused
        for resolution in RESOLUTIONS:
            key = f"k{k}_r{resolution:g}"
            runs = {seed: leiden_partition(fused, n, resolution, seed, "fused") for seed in SEEDS}
            results[key] = runs

    base_runs = results[f"k{BASE_K}_r{BASE_RESOLUTION:g}"]
    labels = base_runs[SEEDS[0]]
    base_edges = edge_cache[BASE_K]
    dino_edges = {edge: {"fused": meta["dino"]} for edge, meta in base_edges.items() if meta["dino"] > 0}
    sig_edges = {edge: {"fused": meta["siglip"]} for edge, meta in base_edges.items() if meta["siglip"] > 0}
    dino_labels = leiden_partition(dino_edges, n, BASE_RESOLUTION, SEEDS[0], "fused")
    sig_labels = leiden_partition(sig_edges, n, BASE_RESOLUTION, SEEDS[0], "fused")
    coords, backbone, extras = make_layout(base_edges, labels, n)

    old_rows = csv_rows(OLD / "canonical_assignments.csv")
    old_by_id = {row["canonical_id"]: row["cluster_fused"] for row in old_rows}
    assignment_rows = []
    for row, record in enumerate(canonical):
        width = int(record["native_width"])
        height = int(record["native_height"])
        if int(record.get("exif_orientation") or 1) in (5, 6, 7, 8):
            width, height = height, width
        assignment_rows.append({
            "row": row,
            "canonical_id": ids[row],
            "source_sha256": record["source_sha256"],
            "source_relative_path": record["source_relative_path"],
            "width": width,
            "height": height,
            "community_leiden_fused": int(labels[row]),
            "community_leiden_dino": int(dino_labels[row]),
            "community_leiden_siglip": int(sig_labels[row]),
            "community_louvain_historical": int(old_by_id[ids[row]]),
            "layout_x": f"{coords[row, 0]:.7f}",
            "layout_y": f"{coords[row, 1]:.7f}",
        })
    fields = list(assignment_rows[0])
    write_csv(OUT / "canonical_assignments.csv", fields, assignment_rows)
    by_id = {row["canonical_id"]: row for row in assignment_rows}
    parent_fields = ["parent_id", "canonical_id", "source_sha256", "source_relative_path", "community_leiden_fused"]
    write_csv(OUT / "parent_assignments.csv", parent_fields, ({
        "parent_id": record["image_id"],
        "canonical_id": record["exact_duplicate_canonical_id"],
        "source_sha256": record["source_sha256"],
        "source_relative_path": record["source_relative_path"],
        "community_leiden_fused": by_id[record["exact_duplicate_canonical_id"]]["community_leiden_fused"],
    } for record in parents))

    with gzip.open(OUT / "photo_edges.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        fields_e = ["source_row", "target_row", "dino_rank_affinity", "siglip_rank_affinity", "fused_affinity", "mutual_dino", "mutual_siglip"]
        writer = csv.DictWriter(stream, fieldnames=fields_e)
        writer.writeheader()
        for (a, b), meta in sorted(base_edges.items()):
            writer.writerow({"source_row": a, "target_row": b,
                             "dino_rank_affinity": f"{float(meta['dino']):.8f}",
                             "siglip_rank_affinity": f"{float(meta['siglip']):.8f}",
                             "fused_affinity": f"{float(meta['fused']):.8f}",
                             "mutual_dino": int(bool(meta["mutual_dino"])),
                             "mutual_siglip": int(bool(meta["mutual_siglip"]))})
    write_csv(OUT / "display_edges.csv", ["source_row", "target_row", "kind"],
              ({"source_row": a, "target_row": b, "kind": "maximum_spanning_forest"} for a, b in backbone))
    with (OUT / "display_edges.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_row", "target_row", "kind"])
        writer.writerows({"source_row": a, "target_row": b, "kind": "mutual_neighbor_extra"} for a, b in extras)

    aggregate = defaultdict(lambda: {"edge_count": 0, "weight_sum": 0.0})
    for (a, b), meta in base_edges.items():
        ca, cb = int(labels[a]), int(labels[b])
        key = tuple(sorted((ca, cb)))
        aggregate[key]["edge_count"] += 1
        aggregate[key]["weight_sum"] += float(meta["fused"])
    write_csv(OUT / "community_edges.csv", ["community_a", "community_b", "edge_count", "weight_sum"],
              ({"community_a": a, "community_b": b, "edge_count": value["edge_count"],
                "weight_sum": f"{value['weight_sum']:.8f}"} for (a, b), value in sorted(aggregate.items())))

    disconnected_count, disconnected_labels = connected_communities(base_edges, labels, n)
    sensitivity = {}
    for key, runs in results.items():
        anchor = runs[SEEDS[0]]
        sensitivity[key] = {
            "communities": int(len(set(anchor.tolist()))),
            "seed_ari": {str(seed): float(adjusted_rand_score(anchor, runs[seed])) for seed in SEEDS[1:]},
            "ari_to_base": float(adjusted_rand_score(labels, anchor)),
        }
    overlap = np.mean([len(set(a[:BASE_K]) & set(b[:BASE_K])) / BASE_K for a, b in zip(dino_nn, sig_nn)])
    metrics = {
        "status": "complete_unsupervised_photo_graph",
        "parents": len(parents),
        "canonicals": n,
        "inputs": {
            "manifest_sha256": sha256(MANIFEST),
            "dinov3_features_sha256": sha256(DINO / "features.npy"),
            "siglip2_features_sha256": sha256(SIGLIP / "features.npy"),
        },
        "base": {"k": BASE_K, "resolution": BASE_RESOLUTION, "seed": SEEDS[0],
                 "algorithm": "Leiden RBConfigurationVertexPartition",
                 "fusion": "0.5*(W_D+W_S)+0.5*min(W_D,W_S)",
                 "communities": int(len(set(labels.tolist()))),
                 "sizes": [count for _, count in Counter(labels.tolist()).most_common()],
                 "edges": len(base_edges),
                 "connected_community_failures": disconnected_count,
                 "disconnected_community_ids": disconnected_labels,
                 "backbone_edges": len(backbone), "mutual_extra_edges": len(extras)},
        "views": {"dino_communities": int(len(set(dino_labels.tolist()))),
                  "siglip_communities": int(len(set(sig_labels.tolist()))),
                  "partition_ari": float(adjusted_rand_score(dino_labels, sig_labels)),
                  "neighbor_overlap_mean_at_30": float(overlap),
                  "leiden_vs_historical_louvain_ari": float(adjusted_rand_score(
                      labels, np.asarray([int(old_by_id[x]) for x in ids], dtype=np.int32)))},
        "sensitivity": sensitivity,
        "layout": "force-directed community quotient and force-directed induced subgraphs; display only",
        "limits": "Photo communities organize appearance. They do not identify components, physical ownership, hidden content, or safety. Region-family edges are added in a separate stage.",
    }
    pages = review_sheets(canonical, np.concatenate([dino, siglip], axis=1), labels)
    metrics["review_pages"] = pages
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    minimum_seed_ari = min(
        metrics["sensitivity"][f"k{BASE_K}_r{BASE_RESOLUTION:g}"]["seed_ari"].values()
    )
    if os.environ.get("RESEARCH_RUN_DIR"):
        run_dir = Path(os.environ["RESEARCH_RUN_DIR"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps({
            "minimum_seed_ari": minimum_seed_ari,
            "communities": metrics["base"]["communities"],
            "connected_community_failures": disconnected_count,
            "photo_edges": len(base_edges),
            "artifact_metrics": str(OUT / "metrics.json"),
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "communities": metrics["base"]["communities"],
                      "sizes": metrics["base"]["sizes"], "edges": len(base_edges),
                      "connected_failures": disconnected_count, "minimum_seed_ari": minimum_seed_ari,
                      "review_pages": len(pages)}))


if __name__ == "__main__":
    main()
