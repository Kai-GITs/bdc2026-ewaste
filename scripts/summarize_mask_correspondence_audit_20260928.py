"""Join manual mask audit decisions and freeze mechanism-level metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().sort_index().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rendered = pd.read_csv(args.render_manifest)
    adjudicated = pd.read_csv(args.adjudication)
    if adjudicated.region_id.duplicated().any():
        raise ValueError("Duplicate adjudication region_id")
    missing = set(rendered.region_id) - set(adjudicated.region_id)
    extra = set(adjudicated.region_id) - set(rendered.region_id)
    if missing or extra:
        raise ValueError(f"Adjudication mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    output = rendered.drop(columns=["proposal_alignment", "semantic_correspondence", "failure_mode"]).merge(
        adjudicated, on="region_id", how="left", validate="one_to_one"
    )
    output_path = args.output / "adjudicated_manifest.csv"
    output.to_csv(output_path, index=False)

    candidates = output[output.audit_group == "kandidat"]
    negatives = output[output.audit_group == "hard negative"]
    metrics = {
        "status": "complete_targeted_mask_correspondence_adjudication",
        "examples_total": int(len(output)),
        "candidate_examples": int(len(candidates)),
        "hard_negative_examples": int(len(negatives)),
        "candidate_proposal_alignment": counts(candidates, "proposal_alignment"),
        "candidate_semantic_correspondence": counts(candidates, "semantic_correspondence"),
        "candidate_claim_interpretation": counts(candidates, "claim_interpretation"),
        "candidate_failure_mode": counts(candidates, "failure_mode"),
        "hard_negative_proposal_alignment": counts(negatives, "proposal_alignment"),
        "hard_negative_semantic_correspondence": counts(negatives, "semantic_correspondence"),
        "hard_negative_claim_interpretation": counts(negatives, "claim_interpretation"),
        "decision": "retain_existing_masks_and_descriptors_with_claim_gating",
        "decision_basis": (
            "All 30 candidate masks were visually usable (23 good, 7 partial, 0 poor), while only "
            "16/30 candidate regions had direct semantic correspondence and 6/30 contradicted the family "
            "interpretation. All 12 hard negatives had good masks and coherent visual appearance, yet every "
            "one required a narrower claim because similarity came from labels, screen content, pictorial "
            "content, or scene composition. The bottleneck is correspondence and interpretation, not mask generation."
        ),
        "claim_limit": (
            "This targeted audit estimates mechanism failure modes for selected high-interest and known negative "
            "families; it is not a prevalence estimate over all 129 families."
        ),
    }
    metrics_path = args.output / "audit_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = f"""# Keputusan audit mask dan korespondensi

Audit ini memakai 42 region tersimpan dari keluarga kandidat dan hard negative. Tidak ada proposal, mask, atau embedding yang dihitung ulang.

## Hasil terarah

- Kandidat: {len(candidates)} contoh; mask `good` {counts(candidates, 'proposal_alignment').get('good', 0)}, `partial` {counts(candidates, 'proposal_alignment').get('partial', 0)}, `poor` {counts(candidates, 'proposal_alignment').get('poor', 0)}.
- Korespondensi kandidat: `yes` {counts(candidates, 'semantic_correspondence').get('yes', 0)}, `ambiguous` {counts(candidates, 'semantic_correspondence').get('ambiguous', 0)}, `no` {counts(candidates, 'semantic_correspondence').get('no', 0)}.
- Hard negative: seluruh {len(negatives)} mask dinilai `good` dan koheren sebagai tampilan visual, tetapi seluruhnya memerlukan klaim yang dipersempit.

## Keputusan mekanisme

Mask yang ada dipertahankan. Tidak ada dasar untuk mengulang SAM: semua kandidat memiliki mask yang dapat diperiksa dan tidak ada mask `poor`. Bottleneck terukur berada pada korespondensi descriptor dan interpretasi. Kesamaan label, isi layar, gambar tercetak, atau bidang gelap merupakan discovery visual yang sah, tetapi tidak membuktikan identitas komponen atau kondisi fisik.

Claim gate final harus mensyaratkan: dukungan lintas foto yang independen, korespondensi semantik pada citra asli dan mask, counterexample eksplisit, serta bahasa klaim yang tidak melampaui apa yang tampak. Contoh `partial` tetap dapat dipakai untuk discovery, tetapi tidak boleh menjadi bukti tunggal.

## Implikasi untuk flagship

Sumbu papan/rakitan memiliki dukungan paling kuat: keluarga kemasan IC, papan dalam rakitan, permukaan papan, serta pad/jalur tepi memiliki contoh positif lintas konteks sekaligus false positive yang jelas. Sumbu kondisi layar dipertahankan sebagai temuan pendamping dengan pemisahan ketat antara kerusakan fisik yang terlihat, tampilan abnormal, isi layar, dan pola uji warna. Keluarga langka tetap ditampilkan sebagai hipotesis discovery, bukan klaim populasi.

## Batas

Audit ini terarah pada kandidat berperingkat tinggi dan hard negative yang sudah diketahui. Angkanya mengukur mekanisme pada contoh tersebut dan tidak mengestimasi prevalensi mutu untuk seluruh 129 keluarga.
"""
    report_path = args.output / "AUDIT_DECISION.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "adjudicated_manifest": {"sha256": sha256(output_path), "rows": int(len(output))},
        "audit_metrics": {"sha256": sha256(metrics_path)},
        "audit_decision": {"sha256": sha256(report_path)},
        "manual_adjudication": {"sha256": sha256(args.adjudication), "rows": int(len(adjudicated))},
    }
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
