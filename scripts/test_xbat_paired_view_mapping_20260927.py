"""Specimen-safe development test of RGB-to-radiograph feature mapping.

The 91 source test specimens have been inspected in earlier analyses, so these
results are descriptive rather than an untouched estimate. All targets are
image-derived vectors; no semantic or component labels are imported.
"""
import csv
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

PROJECT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--feature-dir', type=Path, required=True)
parser.add_argument('--pair-key', type=Path, default=PROJECT / 'experiments/sensor_bridge_20260927/paired_group_key.csv')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
FEATURE = args.feature_dir
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
index = json.loads((FEATURE / "feature_index.json").read_text())
raw = np.load(FEATURE / "xbat_cls.npy", mmap_mode="r")
lookup = {r["id"]: r["index"] for r in index["xbat"]}
with args.pair_key.open(encoding="utf-8", newline="") as f:
    key = list(csv.DictReader(f))
train = sorted(r["specimen_id"] for r in key if r["split"] == "train")
test = sorted(r["specimen_id"] for r in key if r["split"] == "test")
assert len(train) == 330 and len(test) == 91 and not set(train) & set(test)

def vector(ids, modality):
    x = np.stack([raw[lookup[f"{ident}:{modality}"]] for ident in ids]).astype("float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

xtr, ytr = vector(train, "RGB"), vector(train, "HQ")
xte, yte = vector(test, "RGB"), vector(test, "HQ")
alphas = [0.001, 0.01, 0.1, 1., 10., 100.]
folds = list(KFold(n_splits=5, shuffle=True, random_state=270927).split(xtr))
cv = {}
for alpha in alphas:
    scores = []
    for ti, vi in folds:
        m = Ridge(alpha=alpha).fit(xtr[ti], ytr[ti])
        p = norm(m.predict(xtr[vi]))
        scores.extend(np.sum(p*ytr[vi], axis=1).tolist())
    cv[str(alpha)] = float(np.mean(scores))
chosen = max(alphas, key=lambda a: cv[str(a)])
ridge = norm(Ridge(alpha=chosen).fit(xtr, ytr).predict(xte))

sim = xte @ xtr.T
rank = np.argsort(-sim, axis=1)
knn1 = ytr[rank[:, 0]]
knn5 = norm(np.stack([(sim[i, rank[i, :5], None] * ytr[rank[i, :5]]).sum(axis=0)
                       for i in range(len(test))]))
prior = np.repeat(norm(ytr.mean(axis=0))[None, :], len(test), axis=0)

rng = np.random.default_rng(270928)
shuffle_scores = []
for _ in range(50):
    shuffled = ytr[rng.permutation(len(train))]
    p = norm(Ridge(alpha=chosen).fit(xtr, shuffled).predict(xte))
    shuffle_scores.append(float(np.mean(np.sum(p*yte, axis=1))))

def metrics(pred):
    paired = np.sum(pred*yte, axis=1)
    gallery = pred @ yte.T
    order = np.argsort(-gallery, axis=1)
    recall1 = float(np.mean(order[:, 0] == np.arange(len(test))))
    ranks = np.array([int(np.where(order[i] == i)[0][0])+1 for i in range(len(test))])
    return {"paired_hq_cosine_mean": float(paired.mean()),
            "paired_hq_cosine_median": float(np.median(paired)),
            "same_specimen_hq_recall_at_1_of_91": recall1,
            "same_specimen_hq_median_rank_of_91": float(np.median(ranks))}

predictions = {"prior_hq_mean": prior, "rgb_nearest_1_transfers_hq": knn1,
               "rgb_nearest_5_weighted_hq": knn5,
               "paired_ridge_rgb_to_hq": ridge}
paired_scores = {name: np.sum(pred*yte, axis=1) for name, pred in predictions.items()}
with (OUT / "xbat_paired_view_mapping_test_rows.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["specimen_id"] + list(paired_scores))
    writer.writeheader()
    for i, ident in enumerate(test):
        writer.writerow({"specimen_id": ident, **{name:float(values[i])
                                                  for name,values in paired_scores.items()}})

def paired_delta(a, b, seed):
    delta = paired_scores[a] - paired_scores[b]
    draw = np.random.default_rng(seed).integers(0,len(delta),size=(10000,len(delta)))
    means = delta[draw].mean(axis=1)
    return {"mean":float(delta.mean()),"median":float(np.median(delta)),
            "positive_fraction":float(np.mean(delta>0)),
            "bootstrap_mean_interval_95_descriptive":np.quantile(means,[.025,.975]).tolist()}

result = {
    "input": "frozen DINOv2 whole-frame RGB and HQ features, 768 dimensions",
    "split": "original 330 train, 91 test specimen IDs; test previously explored",
    "train_only_cv_mean_paired_cosine_by_ridge_alpha": cv,
    "selected_alpha": chosen,
    "matched_comparisons": {
        "prior_hq_mean": metrics(prior),
        "rgb_nearest_1_transfers_hq": metrics(knn1),
        "rgb_nearest_5_weighted_hq": metrics(knn5),
        "paired_ridge_rgb_to_hq": metrics(ridge),
    },
    "paired_test_query_differences": {
        "ridge_minus_knn5": paired_delta("paired_ridge_rgb_to_hq","rgb_nearest_5_weighted_hq",270929),
        "ridge_minus_knn1": paired_delta("paired_ridge_rgb_to_hq","rgb_nearest_1_transfers_hq",270930),
        "ridge_minus_prior": paired_delta("paired_ridge_rgb_to_hq","prior_hq_mean",270931),
    },
    "shuffled_pair_ridge_50_mean_test_cosine": {
        "mean": float(np.mean(shuffle_scores)),
        "range": [float(min(shuffle_scores)), float(max(shuffle_scores))],
    },
    "interpretation_limit": "HQ embeddings include scanner/framing and are not physical part labels; no BDC X-ray pairing or hidden-content validation",
}
(OUT / "xbat_paired_view_mapping_summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result))
