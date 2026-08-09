"""
Retarget production decision threshold (no model retrain).

Ethical protocol
----------------
* Probabilities come from the already-trained pipelines in the bundle.
* Threshold is chosen on VALIDATION only (same stratified split as training).
* TEST is scored once after threshold is frozen — never used for selection.
* Optional comparison policies are each selected on VAL then evaluated on TEST.

Usage
-----
  .\\.venv\\Scripts\\python.exe scripts\\retarget_threshold.py --strategy middle_ground
  .\\.venv\\Scripts\\python.exe scripts\\retarget_threshold.py --strategy max_accuracy
  .\\.venv\\Scripts\\python.exe scripts\\retarget_threshold.py --strategy youden
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.config import (  # noqa: E402
    APPLICATION_TRAIN,
    FULL_ARTIFACTS_PATH,
    FULL_METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from credit_risk.data import load_application_train, split_features_target  # noqa: E402
from credit_risk.evaluation import full_model_report, ranking_metrics  # noqa: E402
from credit_risk.features_multitable import add_application_derived_features  # noqa: E402
from credit_risk.train import (  # noqa: E402
    VAL_SIZE,
    _evaluate,
    select_threshold,
)

COMPARE_STRATEGIES = ("youden", "f1", "fixed", "middle_ground", "max_accuracy")


def _predict_proba(pipe, frame: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(frame)[:, 1]


def _policy_test_row(
    name: str,
    y_test: np.ndarray,
    proba: np.ndarray,
    thr_info: dict[str, Any],
) -> dict[str, Any]:
    thr = float(thr_info["threshold"])
    rep = full_model_report(name, y_test, proba, thr)
    op = rep["operating_point"]
    return {
        "strategy": thr_info.get("strategy", name),
        "threshold_selected_on": "validation",
        "threshold": thr,
        "validation_snapshot": {
            k: thr_info.get(k)
            for k in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "selection_mode",
                "constraints",
            )
            if k in thr_info
        },
        "test": {
            "accuracy": op.get("accuracy"),
            "balanced_accuracy": op.get("balanced_accuracy"),
            "precision": op["precision"],
            "recall": op["recall"],
            "f1": op["f1"],
            "lift_vs_base_rate": op["lift_vs_base_rate"],
            "flagged_rate": op["flagged_rate"],
            "tp": op["tp"],
            "fp": op["fp"],
            "tn": op["tn"],
            "fn": op["fn"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget threshold ethically (val-only selection)")
    parser.add_argument(
        "--strategy",
        default="middle_ground",
        choices=["youden", "f1", "fixed", "max_accuracy", "middle_ground"],
    )
    parser.add_argument("--artifacts", type=Path, default=FULL_ARTIFACTS_PATH)
    parser.add_argument("--metrics", type=Path, default=FULL_METRICS_PATH)
    args = parser.parse_args()

    artifacts_path: Path = args.artifacts
    metrics_path: Path = args.metrics
    if not artifacts_path.exists():
        raise SystemExit(f"Missing artifacts: {artifacts_path}")
    if not APPLICATION_TRAIN.exists():
        raise SystemExit(f"Missing data: {APPLICATION_TRAIN}")

    print(f"Loading bundle: {artifacts_path}")
    bundle = joblib.load(artifacts_path)
    production = bundle.get("production_model", "xgboost")
    feature_columns = list(bundle["feature_columns"])
    multi_table = bool(bundle.get("multi_table", False))
    if multi_table:
        raise SystemExit("Application-only packs supported by this script.")

    print("Loading application_train (full)...")
    df = load_application_train(sample_size=None)
    df = add_application_derived_features(df)
    X, y = split_features_target(df)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_columns]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    print(f"Split train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    lr_pipe = bundle["logistic_regression"]
    xgb_pipe = bundle["xgboost"]
    dummy_pipe = bundle["dummy_stratified"]
    prod_pipe = xgb_pipe if production == "xgboost" else lr_pipe

    print("Scoring validation / test (no fitting)...")
    prod_val = _predict_proba(prod_pipe, X_val)
    thr_info = select_threshold(y_val.values, prod_val, strategy=args.strategy)
    thr_info["selected_on"] = "validation"
    threshold = float(thr_info["threshold"])
    print(
        f"Production threshold ({args.strategy} on val): tau={threshold:.4f} "
        f"val_acc={thr_info.get('accuracy', float('nan')):.4%} "
        f"val_rec={thr_info.get('recall', float('nan')):.4%} "
        f"val_f1={thr_info.get('f1', float('nan')):.4f}"
    )

    lr_test = _predict_proba(lr_pipe, X_test)
    xgb_test = _predict_proba(xgb_pipe, X_test)
    dummy_test = _predict_proba(dummy_pipe, X_test)
    prod_test = xgb_test if production == "xgboost" else lr_test

    # Fair policy comparison: each strategy chosen on VAL, metrics on TEST
    policy_comparison: dict[str, Any] = {}
    for strat in COMPARE_STRATEGIES:
        info = select_threshold(y_val.values, prod_val, strategy=strat)
        info["selected_on"] = "validation"
        policy_comparison[strat] = _policy_test_row(strat, y_test.values, prod_test, info)

    n_train, n_test = len(X_train), len(X_test)
    dr_train, dr_test = float(y_train.mean()), float(y_test.mean())

    dummy_metrics = _evaluate(
        "dummy_stratified", y_test.values, dummy_test, n_train, n_test, dr_train, dr_test, threshold
    )
    lr_metrics = _evaluate(
        "logistic_regression", y_test.values, lr_test, n_train, n_test, dr_train, dr_test, threshold
    )
    xgb_metrics = _evaluate(
        "xgboost", y_test.values, xgb_test, n_train, n_test, dr_train, dr_test, threshold
    )

    lr_report = full_model_report("logistic_regression", y_test.values, lr_test, threshold)
    xgb_report = full_model_report("xgboost", y_test.values, xgb_test, threshold)
    dummy_report = full_model_report("dummy_stratified", y_test.values, dummy_test, threshold)
    lr_rank = ranking_metrics(y_test.values, lr_test)
    xgb_rank = ranking_metrics(y_test.values, xgb_test)
    dummy_rank = ranking_metrics(y_test.values, dummy_test)

    prod_metrics = xgb_metrics if production == "xgboost" else lr_metrics
    prod_op = xgb_report["operating_point"] if production == "xgboost" else lr_report["operating_point"]
    print(
        f"TEST ({production}): accuracy={prod_metrics.accuracy:.4%} "
        f"bal_acc={prod_metrics.balanced_accuracy:.4%} "
        f"precision={prod_metrics.precision:.4%} "
        f"recall={prod_metrics.recall:.4%} "
        f"f1={prod_op['f1']:.4f} "
        f"roc_auc={prod_metrics.roc_auc:.4f}"
    )

    old_metrics: dict = {}
    if metrics_path.exists():
        old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    metrics_payload = {
        "lifecycle": old_metrics.get("lifecycle", "academic_v2_train_val_test"),
        "stages": old_metrics.get("stages", []),
        "hyperparameter_tuning": old_metrics.get("hyperparameter_tuning"),
        "production_model": production,
        "threshold": threshold,
        "threshold_info": thr_info,
        "sample_size": old_metrics.get("sample_size"),
        "n_rows_used": int(len(X)),
        "n_features": int(len(feature_columns)),
        "multi_table": multi_table,
        "split": {
            "n_train": int(len(X_train)),
            "n_validation": int(len(X_val)),
            "n_test": int(len(X_test)),
            "threshold_selected_on": "validation",
            "model_selected_on": "validation",
            "final_metrics_on": "test",
        },
        "base_rate_test": float(y_test.mean()),
        "validation_ranking": old_metrics.get("validation_ranking"),
        "models": {
            "dummy_stratified": {
                **asdict(dummy_metrics),
                "ranking": dummy_rank,
                "operating_point": dummy_report["operating_point"],
            },
            "logistic_regression": {
                **asdict(lr_metrics),
                "ranking": lr_rank,
                "operating_point": lr_report["operating_point"],
            },
            "xgboost": {
                **asdict(xgb_metrics),
                "ranking": xgb_rank,
                "operating_point": xgb_report["operating_point"],
            },
        },
        "threshold_sweep": {
            "logistic_regression": lr_report["threshold_sweep"],
            "xgboost": xgb_report["threshold_sweep"],
        },
        "policy_comparison": {
            "protocol": (
                "Each named strategy selects tau on VALIDATION production scores; "
                "test metrics below are evaluated after freezing that tau. "
                "Models were not refit."
            ),
            "production_strategy": args.strategy,
            "strategies": policy_comparison,
        },
        "beats_baseline": {
            "logistic_regression_roc_auc_gt_dummy": lr_metrics.roc_auc
            > dummy_metrics.roc_auc + 0.05,
            "xgboost_roc_auc_gt_dummy": xgb_metrics.roc_auc > dummy_metrics.roc_auc + 0.05,
            "xgboost_accuracy_ge_0_80": xgb_metrics.accuracy >= 0.80,
            "xgboost_recall_ge_0_35": xgb_metrics.recall >= 0.35,
        },
        "skeptics_summary": {
            "primary_reported_metrics": [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ],
            "threshold_policy": f"{args.strategy} on validation",
            "headline_test_accuracy": float(prod_metrics.accuracy),
            "headline_test_recall": float(prod_metrics.recall),
            "headline_test_precision": float(prod_metrics.precision),
            "headline_test_balanced_accuracy": float(prod_metrics.balanced_accuracy),
            "roc_auc_unchanged_by_threshold": True,
            "ethical_notes": [
                "No test labels used for threshold selection",
                "No labels fabricated; no train/test leakage",
                "Only the decision threshold changed; model weights unchanged",
            ],
        },
        "note": (
            f"Threshold strategy={args.strategy} selected on VALIDATION only "
            f"(tau={threshold:.4f}). "
            f"Test accuracy={prod_metrics.accuracy:.2%}, "
            f"recall={prod_metrics.recall:.2%}, "
            f"precision={prod_metrics.precision:.2%}, "
            f"balanced_accuracy={prod_metrics.balanced_accuracy:.2%}, "
            f"ROC-AUC={prod_metrics.roc_auc:.4f}. "
            "ROC-AUC/PR-AUC are threshold-independent. "
            "Decline if P(default) >= threshold."
        ),
        "retarget": {
            "previous_threshold": old_metrics.get("threshold"),
            "previous_strategy": (old_metrics.get("threshold_info") or {}).get("strategy"),
            "new_threshold": threshold,
            "strategy": args.strategy,
            "models_retrained": False,
            "methodology_doc": "docs/OPERATING_POLICY.md",
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Wrote metrics -> {metrics_path}")

    # Markdown summary
    pc_lines = [
        "| Policy (val-selected) | τ | Test Acc | Bal Acc | Precision | Recall | F1 | Lift |",
        "|-----------------------|---|----------|---------|-----------|--------|----|------|",
    ]
    for strat in COMPARE_STRATEGIES:
        row = policy_comparison[strat]
        t = row["test"]
        star = " **(production)**" if strat == args.strategy else ""
        pc_lines.append(
            f"| {strat}{star} | {row['threshold']:.3f} | "
            f"{100*t['accuracy']:.2f}% | {100*t['balanced_accuracy']:.2f}% | "
            f"{100*t['precision']:.2f}% | {100*t['recall']:.2f}% | "
            f"{t['f1']:.3f} | {t['lift_vs_base_rate']:.2f}x |"
        )

    summary_path = metrics_path.parent / "metrics_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Model metrics summary (FULL DATA — {args.strategy.upper()} THRESHOLD)",
                "",
                f"- Production model: **{production}**",
                f"- Rows used: **{len(X)}**",
                f"- Threshold strategy: **{args.strategy}** (chosen on **validation** only)",
                f"- Threshold τ: **{threshold:.4f}**",
                f"- Final metrics on: **test**",
                f"- Test base rate: **{float(y_test.mean()):.2%}**",
                f"- Models retrained for this retarget: **no**",
                f"- Methodology: [`docs/OPERATING_POLICY.md`](../../docs/OPERATING_POLICY.md)",
                "",
                "## Headline (production operating point — test)",
                "",
                "| Metric | XGBoost (test) |",
                "|--------|----------------|",
                f"| **Accuracy** | **{xgb_metrics.accuracy:.2%}** |",
                f"| **Balanced accuracy** | **{xgb_metrics.balanced_accuracy:.2%}** |",
                f"| **Precision** | **{xgb_metrics.precision:.2%}** |",
                f"| **Recall** | **{xgb_metrics.recall:.2%}** |",
                f"| F1 | {prod_op['f1']:.4f} |",
                f"| ROC-AUC | {xgb_metrics.roc_auc:.4f} |",
                f"| PR-AUC | {xgb_metrics.pr_auc:.4f} |",
                f"| Lift@τ | {prod_op['lift_vs_base_rate']:.2f}x |",
                "",
                "## All models @ production τ",
                "",
                "| Model | Accuracy | Bal. Acc | ROC-AUC | Precision | Recall | Lift |",
                "|-------|----------|----------|---------|-----------|--------|------|",
                (
                    f"| Dummy | {dummy_metrics.accuracy:.4f} | {dummy_metrics.balanced_accuracy:.4f} | "
                    f"{dummy_metrics.roc_auc:.4f} | {dummy_metrics.precision:.4f} | "
                    f"{dummy_metrics.recall:.4f} | "
                    f"{dummy_report['operating_point']['lift_vs_base_rate']:.2f}x |"
                ),
                (
                    f"| Logistic Regression | {lr_metrics.accuracy:.4f} | {lr_metrics.balanced_accuracy:.4f} | "
                    f"{lr_metrics.roc_auc:.4f} | {lr_metrics.precision:.4f} | "
                    f"{lr_metrics.recall:.4f} | "
                    f"{lr_report['operating_point']['lift_vs_base_rate']:.2f}x |"
                ),
                (
                    f"| **XGBoost** | **{xgb_metrics.accuracy:.4f}** | **{xgb_metrics.balanced_accuracy:.4f}** | "
                    f"**{xgb_metrics.roc_auc:.4f}** | **{xgb_metrics.precision:.4f}** | "
                    f"**{xgb_metrics.recall:.4f}** | "
                    f"**{xgb_report['operating_point']['lift_vs_base_rate']:.2f}x** |"
                ),
                "",
                "## Policy comparison (same model; each τ from validation)",
                "",
                *pc_lines,
                "",
                f"Operating point (test): TP={prod_op['tp']} FP={prod_op['fp']} "
                f"TN={prod_op['tn']} FN={prod_op['fn']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary -> {summary_path}")

    bundle["threshold"] = threshold
    bundle["threshold_info"] = thr_info
    bundle["metrics"] = {
        "dummy_stratified": asdict(dummy_metrics),
        "logistic_regression": asdict(lr_metrics),
        "xgboost": asdict(xgb_metrics),
    }
    joblib.dump(bundle, artifacts_path)
    print(f"Updated bundle -> {artifacts_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
