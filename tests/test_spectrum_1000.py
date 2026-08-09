"""
1000-case spectrum suite: every applicant must pass decision logic + reasoning checks.

Uses the production full-data bundle when present; otherwise the session-trained
sample bundle (same assess_applicant path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from credit_risk.config import FULL_ARTIFACTS_PATH
from credit_risk.predict import (
    assess_applicant,
    load_bundle,
    probability_to_decision,
    reset_bundle_cache,
)

from spectrum_generator import features_only, generate_spectrum_cases

N_CASES = 1000
SEED = 42


@pytest.fixture(scope="module")
def spectrum_bundle():
    """Prefer production pack; fall back to small trained fixture path via conftest."""
    reset_bundle_cache()
    if FULL_ARTIFACTS_PATH.exists():
        b = load_bundle(path=FULL_ARTIFACTS_PATH, force_reload=True)
        yield b
        reset_bundle_cache()
        return
    # Fallback: train small model once (via session fixture pattern)
    from credit_risk.config import APPLICATION_TRAIN
    from credit_risk.train import train_models

    if not APPLICATION_TRAIN.exists():
        pytest.skip("No production bundle and no Home Credit data")
    tmp = Path(__file__).resolve().parents[1] / ".pytest_spectrum_bundle"
    tmp.mkdir(exist_ok=True)
    art = tmp / "credit_risk_bundle.joblib"
    met = tmp / "metrics.json"
    if not art.exists():
        train_models(
            sample_size=4000,
            artifacts_path=art,
            metrics_path=met,
            preferred_model="xgboost",
            run_eda=False,
            threshold_strategy="middle_ground",
        )
    b = load_bundle(path=art, force_reload=True)
    yield b
    reset_bundle_cache()


@pytest.fixture(scope="module")
def spectrum_cases():
    return generate_spectrum_cases(N_CASES, seed=SEED, include_anchors=True)


@pytest.fixture(scope="module")
def spectrum_results(spectrum_bundle, spectrum_cases):
    """Score all 1000 once (production model)."""
    rows = []
    for case in spectrum_cases:
        feats = features_only(case)
        out = assess_applicant(feats, bundle=spectrum_bundle)
        rows.append(
            {
                "case_id": case["case_id"],
                "risk_level": case["risk_level"],
                "decision": out["decision"],
                "default_probability": float(out["default_probability"]),
                "risk_score": float(out["risk_score"]),
                "threshold": float(out["threshold"]),
                "model": out["model"],
                "message": out.get("message", ""),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Per-case logic (all 1000)
# ---------------------------------------------------------------------------


def test_all_1000_cases_pass_decision_logic(spectrum_results):
    """Every case: valid proba, decision, threshold rule, message, aliases."""
    assert len(spectrum_results) == N_CASES
    failures: list[str] = []

    for r in spectrum_results:
        cid = r["case_id"]
        p = r["default_probability"]
        thr = r["threshold"]
        dec = r["decision"]

        if not (0.0 <= p <= 1.0):
            failures.append(f"case {cid}: proba {p} out of [0,1]")
        if dec not in {"approve", "decline"}:
            failures.append(f"case {cid}: bad decision {dec!r}")
        expected = probability_to_decision(p, thr)
        if dec != expected:
            failures.append(
                f"case {cid}: decision {dec} != rule({p:.6f}, τ={thr:.4f})={expected}"
            )
        if abs(r["risk_score"] - p) > 1e-12:
            failures.append(f"case {cid}: risk_score != default_probability")
        if not r["model"]:
            failures.append(f"case {cid}: empty model name")
        if dec == "decline" and "declined" not in r["message"].lower():
            failures.append(f"case {cid}: decline message missing")
        if dec == "approve" and "approved" not in r["message"].lower():
            failures.append(f"case {cid}: approve message missing")

    assert not failures, (
        f"{len(failures)} / {N_CASES} logic failures (showing up to 15):\n"
        + "\n".join(failures[:15])
    )


def test_all_1000_both_models_obey_threshold(spectrum_bundle, spectrum_cases):
    """XGBoost and Logistic Regression both respect decline iff p >= τ."""
    models = ["xgboost", "logistic_regression"]
    failures: list[str] = []
    # Subsample density but still 1000 total evaluations (500 each) would be less;
    # user asked for a thousand cases — score all 1000 on production model already;
    # for dual-model, score all 1000 on both (2000 total) for full coverage.
    for model in models:
        for case in spectrum_cases:
            out = assess_applicant(
                features_only(case), bundle=spectrum_bundle, model_name=model
            )
            p = float(out["default_probability"])
            thr = float(out["threshold"])
            exp = probability_to_decision(p, thr)
            if out["decision"] != exp:
                failures.append(
                    f"{model} case {case['case_id']}: {out['decision']} vs {exp} (p={p:.4f})"
                )
            if not (0.0 <= p <= 1.0):
                failures.append(f"{model} case {case['case_id']}: bad p={p}")
            if out["model"] != model:
                failures.append(f"{model} case {case['case_id']}: model field {out['model']}")

    assert not failures, (
        f"{len(failures)} dual-model failures (up to 15):\n" + "\n".join(failures[:15])
    )


def test_threshold_override_logic_on_spectrum_sample(spectrum_bundle, spectrum_cases):
    """Forced thr=0 → all decline; thr=1.0 → approve unless p is exactly 1.0."""
    sample = spectrum_cases[::50]  # 20 cases across spectrum
    for case in sample:
        feats = features_only(case)
        low = assess_applicant(feats, bundle=spectrum_bundle, threshold=0.0)
        assert low["decision"] == "decline", case["case_id"]
        high = assess_applicant(feats, bundle=spectrum_bundle, threshold=1.0)
        # decline only if p >= 1.0
        exp = probability_to_decision(high["default_probability"], 1.0)
        assert high["decision"] == exp


# ---------------------------------------------------------------------------
# Spectrum / reasoning (aggregate, still grounded in the 1000 cases)
# ---------------------------------------------------------------------------


def test_spectrum_risk_ordering_reasoning(spectrum_results):
    """
    Designed safer profiles must score lower default risk on average than
    designed riskier profiles. Pure poles: safest < riskiest probability.
    """
    by_id = {r["case_id"]: r for r in spectrum_results}
    safest = by_id[0]["default_probability"]
    riskiest = by_id[N_CASES - 1]["default_probability"]
    assert riskiest > safest + 0.05, (
        f"Risk spectrum inverted or too flat: safest p={safest:.4f}, "
        f"riskiest p={riskiest:.4f}"
    )

    levels = np.array([r["risk_level"] for r in spectrum_results])
    probs = np.array([r["default_probability"] for r in spectrum_results])

    low_mask = levels <= 0.20
    high_mask = levels >= 0.80
    assert low_mask.sum() >= 50 and high_mask.sum() >= 50
    mean_low = float(probs[low_mask].mean())
    mean_high = float(probs[high_mask].mean())
    assert mean_high > mean_low + 0.08, (
        f"Expected high risk_level band to have higher mean P(default): "
        f"low={mean_low:.4f} high={mean_high:.4f}"
    )

    # Rank correlation (Spearman via scipy if available; else manual)
    try:
        from scipy.stats import spearmanr

        corr, pval = spearmanr(levels, probs)
    except Exception:
        # Fallback: compare quartile means monotonic-ish
        qs = []
        for a, b in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            m = (levels >= a) & (levels < b)
            qs.append(float(probs[m].mean()))
        assert qs[-1] > qs[0], f"quartile means not ordered: {qs}"
        corr = 1.0 if qs[-1] > qs[0] else 0.0
        pval = 0.0

    assert corr > 0.35, f"Spearman(risk_level, P(default)) too weak: {corr:.3f}"
    assert pval < 0.01 or corr > 0.5


def test_spectrum_decline_rate_rises_with_risk(spectrum_results):
    """Decline rate in high risk_level band must exceed low band (reasoning check)."""
    levels = np.array([r["risk_level"] for r in spectrum_results])
    dec = np.array([1 if r["decision"] == "decline" else 0 for r in spectrum_results])
    low = dec[levels <= 0.25].mean()
    high = dec[levels >= 0.75].mean()
    assert high > low + 0.10, (
        f"Decline rate should rise with designed risk: low_band={low:.3f} high_band={high:.3f}"
    )


def test_extreme_poles_have_expected_decisions(spectrum_results, spectrum_bundle):
    """
    At production τ, pure safest pole should approve and pure riskiest should decline
    on a well-trained full model. If sample model is weak, still require riskiest
    probability strictly greater and decision consistent with its own p.
    """
    by_id = {r["case_id"]: r for r in spectrum_results}
    safe = by_id[0]
    risky = by_id[N_CASES - 1]
    # Always: internal consistency
    assert safe["decision"] == probability_to_decision(
        safe["default_probability"], safe["threshold"]
    )
    assert risky["decision"] == probability_to_decision(
        risky["default_probability"], risky["threshold"]
    )
    assert risky["default_probability"] > safe["default_probability"]

    # On production full-data pack we expect clear approve/decline poles
    if FULL_ARTIFACTS_PATH.exists() and spectrum_bundle.threshold < 0.9:
        assert safe["decision"] == "approve", (
            f"Safest pole should approve at τ={safe['threshold']}: p={safe['default_probability']}"
        )
        assert risky["decision"] == "decline", (
            f"Riskiest pole should decline at τ={risky['threshold']}: p={risky['default_probability']}"
        )


def test_no_nan_or_inf_probabilities(spectrum_results):
    for r in spectrum_results:
        p = r["default_probability"]
        assert np.isfinite(p), r["case_id"]
