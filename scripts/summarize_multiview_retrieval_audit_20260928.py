"""Join the visually inspected pair audit to retrieval outputs and summarize it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rendered = pd.read_csv(args.render_manifest)
    manual = pd.read_csv(args.adjudication)
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions.arm.eq("dino_foreground_nn_all_r2")].copy()
    if len(rendered) != len(manual) or len(rendered) != len(predictions) or len(rendered) != 42:
        raise ValueError("The pair audit must contain the same 42 queries in every source")
    if set(rendered.query_region_id) != set(manual.query_region_id):
        raise ValueError("Manual adjudication IDs do not match the rendered evidence")
    result = rendered.drop(columns=["pair_semantic_correspondence", "pair_evidence_note"]).merge(
        manual, on="query_region_id", validate="one_to_one"
    ).merge(
        predictions[["query_region_id", "query_visual_name_id", "query_family_internal",
                     "candidate_family_internal", "dino_foreground_cosine", "dino_box_cosine",
                     "siglip2_crop_cosine", "expected_family_rank"]],
        on="query_region_id", validate="one_to_one"
    )
    result.to_csv(args.output, index=False)

    visible = result[result.gold_visible_family.astype(bool)]
    semantic_yes = result.pair_semantic_correspondence.eq("yes")
    supported = result.claim_status.eq("supported")
    same_family = result.same_r2_family.astype(bool)
    metrics = {
        "status": "complete_one_agent_original_photo_pair_adjudication",
        "pairs": int(len(result)),
        "visible_target_pairs": int(len(visible)),
        "pair_semantic_correspondence": result.pair_semantic_correspondence.value_counts().to_dict(),
        "claim_status": result.claim_status.value_counts().to_dict(),
        "same_r2_family_pairs": int(same_family.sum()),
        "same_r2_family_and_semantic_yes": int((same_family & semantic_yes).sum()),
        "visible_top1_semantic_yes": int(visible.pair_semantic_correspondence.eq("yes").sum()),
        "visible_top1_semantic_ambiguous": int(visible.pair_semantic_correspondence.eq("ambiguous").sum()),
        "visible_top1_semantic_no": int(visible.pair_semantic_correspondence.eq("no").sum()),
        "visible_top1_claim_supported": int(visible.claim_status.eq("supported").sum()),
        "visible_top1_claim_supported_fraction": float(visible.claim_status.eq("supported").mean()),
        "strict_supported_all_queries": int(supported.sum()),
        "evaluation_source": "one Codex agent, original-photo plus exact stored mask and crop, 2026-09-28",
        "not_human_operator_evaluation": True,
        "not_population_prevalence": True,
        "limitations": [
            "The 42-query audit is targeted toward high-interest families and known shortcuts.",
            "The r2 partition was created on the same collection before this evaluation.",
            "A family match is accepted only after original-photo pair inspection; semantic judgments still come from one agent.",
            "No operator time, clicks, downstream yield, or new-photo generalization was measured.",
        ],
        "inputs": {
            str(args.render_manifest): digest(args.render_manifest),
            str(args.adjudication): digest(args.adjudication),
            str(args.predictions): digest(args.predictions),
        },
        "output_sha256": digest(args.output),
    }
    metrics_path = args.output.with_name("pair_audit_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
