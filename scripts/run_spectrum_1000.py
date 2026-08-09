"""
Generate & score 1000 spectrum test cases; write a JSON + markdown report.

  .\\.venv\\Scripts\\python.exe scripts\\run_spectrum_1000.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from credit_risk.config import FULL_ARTIFACTS_PATH  # noqa: E402
from credit_risk.predict import (  # noqa: E402
    assess_applicant,
    load_bundle,
    probability_to_decision,
    reset_bundle_cache,
)
from spectrum_generator import features_only, generate_spectrum_cases  # noqa: E402

N = 1000
OUT_DIR = ROOT / "reports" / "spectrum_1000"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FULL_ARTIFACTS_PATH.exists():
        print(f"ERROR: missing {FULL_ARTIFACTS_PATH}")
        return 1

    reset_bundle_cache()
    bundle = load_bundle(path=FULL_ARTIFACTS_PATH, force_reload=True)
    cases = generate_spectrum_cases(N, seed=42, include_anchors=True)

    rows = []
    logic_fail = 0
    for case in cases:
        out = assess_applicant(features_only(case), bundle=bundle)
        p = float(out["default_probability"])
        thr = float(out["threshold"])
        dec = out["decision"]
        exp = probability_to_decision(p, thr)
        ok = (
            dec == exp
            and 0.0 <= p <= 1.0
            and dec in {"approve", "decline"}
            and abs(float(out["risk_score"]) - p) < 1e-12
        )
        if not ok:
            logic_fail += 1
        rows.append(
            {
                "case_id": case["case_id"],
                "risk_level": case["risk_level"],
                "risk_level_effective": case.get("risk_level_effective"),
                "decision": dec,
                "default_probability": p,
                "threshold": thr,
                "expected_decision": exp,
                "logic_ok": ok,
                "model": out["model"],
                "message": out.get("message"),
                "EXT_SOURCE_2": case.get("EXT_SOURCE_2"),
                "AMT_INCOME_TOTAL": case.get("AMT_INCOME_TOTAL"),
                "AMT_CREDIT": case.get("AMT_CREDIT"),
            }
        )

    probs = np.array([r["default_probability"] for r in rows])
    levels = np.array([r["risk_level"] for r in rows])
    declines = np.array([r["decision"] == "decline" for r in rows])

    summary = {
        "n_cases": N,
        "model": bundle.production_model,
        "threshold": float(bundle.threshold),
        "logic_failures": logic_fail,
        "all_logic_passed": logic_fail == 0,
        "n_approve": int((~declines).sum()),
        "n_decline": int(declines.sum()),
        "decline_rate": float(declines.mean()),
        "mean_proba": float(probs.mean()),
        "proba_safest": float(rows[0]["default_probability"]),
        "proba_riskiest": float(rows[-1]["default_probability"]),
        "decision_safest": rows[0]["decision"],
        "decision_riskiest": rows[-1]["decision"],
        "mean_proba_low_band_0_20": float(probs[levels <= 0.20].mean()),
        "mean_proba_high_band_80_100": float(probs[levels >= 0.80].mean()),
        "decline_rate_low_band_0_25": float(declines[levels <= 0.25].mean()),
        "decline_rate_high_band_75_100": float(declines[levels >= 0.75].mean()),
    }

    try:
        from scipy.stats import spearmanr

        corr, pval = spearmanr(levels, probs)
        summary["spearman_risk_level_vs_proba"] = float(corr)
        summary["spearman_pvalue"] = float(pval)
    except Exception:
        summary["spearman_risk_level_vs_proba"] = None

    (OUT_DIR / "spectrum_1000_results.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "spectrum_1000_cases.json").write_text(
        json.dumps(cases, indent=2),
        encoding="utf-8",
    )

    md = [
        "# Spectrum 1000 test report",
        "",
        f"- Cases: **{N}**",
        f"- Model: **{summary['model']}**",
        f"- Threshold τ: **{summary['threshold']:.4f}**",
        f"- Logic failures: **{logic_fail}** → "
        f"{'ALL PASSED' if logic_fail == 0 else 'FAILED'}",
        f"- Approve / Decline: **{summary['n_approve']}** / **{summary['n_decline']}** "
        f"(decline rate {summary['decline_rate']:.1%})",
        f"- Safest pole: p={summary['proba_safest']:.4f} → **{summary['decision_safest']}**",
        f"- Riskiest pole: p={summary['proba_riskiest']:.4f} → **{summary['decision_riskiest']}**",
        f"- Mean P low band (≤0.2): {summary['mean_proba_low_band_0_20']:.4f}",
        f"- Mean P high band (≥0.8): {summary['mean_proba_high_band_80_100']:.4f}",
        f"- Decline rate low (≤0.25): {summary['decline_rate_low_band_0_25']:.1%}",
        f"- Decline rate high (≥0.75): {summary['decline_rate_high_band_75_100']:.1%}",
    ]
    if summary.get("spearman_risk_level_vs_proba") is not None:
        md.append(
            f"- Spearman(risk_level, P): **{summary['spearman_risk_level_vs_proba']:.3f}** "
            f"(p={summary['spearman_pvalue']:.2e})"
        )
    md.append("")
    md.append("Artifacts: `spectrum_1000_results.json`, `spectrum_1000_cases.json`")
    (OUT_DIR / "spectrum_1000_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))
    print(f"\nWrote {OUT_DIR}")
    return 0 if logic_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
