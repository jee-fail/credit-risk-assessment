# Operating policy — decision threshold (ethical middle ground)

**Status:** production  
**Last updated:** after `middle_ground` retarget on full-data XGBoost pack  
**Source of truth for numbers:** `models/full_data/metrics.json` · `models/full_data/metrics_summary.md` · `STATUS.md`

---

## 1. What this document covers

Credit models output a **probability of default** \(P(\text{default})\).  
A **threshold** \(\tau\) turns that score into a business decision:

| Rule | Decision |
|------|----------|
| \(P(\text{default}) \ge \tau\) | **decline** |
| \(P(\text{default}) < \tau\) | **approve** |

Changing \(\tau\) **does not retrain** the model and **does not change ROC-AUC / PR-AUC**.  
It only changes accuracy, precision, recall, and the confusion matrix at the operating point.

---

## 2. Ethical methodology (non-negotiable)

| Rule | How we follow it |
|------|------------------|
| No label leakage | Threshold chosen on **validation only** |
| Test held out | Test metrics computed **after** \(\tau\) is frozen |
| No fabricated labels | All metrics from real Home Credit `TARGET` |
| No silent metric shopping on test | Constraints and objective fixed **a priori** |
| Transparency | Full policy comparison table published |
| Reproducible | Same stratified split (`RANDOM_STATE=42`, `TEST_SIZE=0.2`, `VAL_SIZE=0.2`) |

**Models were not refit** for the middle-ground policy. Only the stored decision threshold in the artifact bundle was updated.

---

## 3. Production policy: `middle_ground`

### 3.1 Motivation

Two extremes are undesirable for review and for risk:

| Extreme | Typical outcome | Problem |
|---------|-----------------|---------|
| Youden (high recall) | Acc ~69%, Recall ~71% | Accuracy looks “low” to non-specialists |
| Max accuracy | Acc ~92%, Recall ~7% | Almost never catches defaulters |

**Middle ground** keeps **accuracy ≥ 80%** while preserving **meaningful recall (≥ 35%)**.

### 3.2 Selection algorithm (validation only)

Fixed a priori (not tuned on test):

1. Scan \(\tau \in \{0.05, 0.06, \ldots, 0.95\}\).
2. Keep candidates with **validation accuracy ≥ 0.80** and **validation recall ≥ 0.35**.
3. Among survivors, **maximize F1** (balances precision and recall).
4. Tie-breakers: balanced accuracy → accuracy → recall.
5. If no candidate is feasible: fall back to maximizing the harmonic mean of accuracy and recall.

Implemented in `credit_risk.train.select_threshold(..., strategy="middle_ground")`.

### 3.3 Production numbers (held-out **test**)

| Field | Value |
|-------|--------|
| Strategy | `middle_ground` |
| \(\tau\) | **0.5100** |
| Accuracy | **86.20%** |
| Balanced accuracy | **65.18%** |
| Precision | **26.54%** |
| Recall | **40.10%** |
| F1 | 0.319 |
| ROC-AUC | **0.769** (threshold-independent) |
| PR-AUC | **0.261** |
| Lift@τ | **3.29×** |
| Specificity | **~90.3%** |
| NPV | **~94.5%** |
| Confusion | TP=1991 · FP=5511 · TN=51027 · FN=2974 |

Interpretation in plain language:

- About **86 of 100** applicants get the correct approve/decline label under this cut-off.
- Of true defaulters, we catch about **40 of 100** (recall).
- Among those we decline, about **27 of 100** are true defaulters (precision) — **~3.3×** the 8% base rate (lift).
- Of those we **approve**, about **94.5%** truly do not default (NPV) — useful for “safety of approvals.”

---

## 4. Policy comparison (same model, fair protocol)

Each row: select \(\tau\) on **validation**, evaluate on **test**.  
Production model probabilities are identical across rows.

| Policy | τ | Test Acc | Bal Acc | Precision | Recall | F1 | Lift |
|--------|---|----------|---------|-----------|--------|----|------|
| youden | 0.303 | 68.98% | 70.01% | 16.69% | 71.24% | 0.271 | 2.07× |
| fixed (0.5) | 0.500 | 85.68% | 65.69% | 25.99% | 41.85% | 0.321 | 3.22× |
| **middle_ground (production)** | **0.510** | **86.20%** | **65.18%** | **26.54%** | **40.10%** | **0.319** | **3.29×** |
| f1 | 0.525 | 86.82% | 64.36% | 27.14% | 37.58% | 0.315 | 3.36× |
| max_accuracy | 0.760 | 91.88% | 53.33% | 48.09% | 7.35% | 0.128 | 5.96× |

This table is also in `models/full_data/metrics.json` → `policy_comparison`.

---

## 5. How to reproduce / re-apply

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"

# Apply middle-ground policy (default) — no retrain
.\.venv\Scripts\python.exe scripts\retarget_threshold.py --strategy middle_ground

# Optional alternatives (also val-only selection)
.\.venv\Scripts\python.exe scripts\retarget_threshold.py --strategy youden
.\.venv\Scripts\python.exe scripts\retarget_threshold.py --strategy max_accuracy
.\.venv\Scripts\python.exe scripts\retarget_threshold.py --strategy f1
```

Full retrain with the same policy from the start:

```powershell
.\.venv\Scripts\python.exe -m credit_risk.train `
  --sample-size 0 --tune --tune-iter 10 --no-eda `
  --threshold-strategy middle_ground `
  --artifacts models/full_data/credit_risk_bundle.joblib `
  --metrics models/full_data/metrics.json
```

---

## 6. What to put on a review slide

**Lead with the balanced block (not a single cherry-picked metric):**

| Metric | Value |
|--------|-------|
| Accuracy | **86.2%** |
| Recall (defaulters caught) | **40.1%** |
| Precision | **26.5%** |
| Balanced accuracy | **65.2%** |
| ROC-AUC | **0.769** |

**One sentence:**  
> We select the approve/decline threshold on validation under explicit floors (accuracy ≥ 80%, recall ≥ 35%), maximize F1 among feasible cut-offs, and report all metrics on a held-out test set. Models were not altered for this policy.

---

## 7. Related files (keep in sync)

| File | Role |
|------|------|
| `models/full_data/credit_risk_bundle.joblib` | Serving threshold + pipelines |
| `models/full_data/metrics.json` | Full JSON metrics + policy comparison |
| `models/full_data/metrics_summary.md` | Human-readable table |
| `models/full_data/README.md` | Pack summary |
| `STATUS.md` | Project status source of truth |
| `README.md` | Top-level results |
| `docs/MODEL_CARD.md` | Model card |
| `src/credit_risk/config.py` | `PROJECT_STATUS` constants |
| `dashboard/app.py` | Displays metrics from `metrics.json` |
| `scripts/retarget_threshold.py` | Ethical retarget utility |
