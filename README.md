# Credit Risk Assessment System

**Academic end-to-end ML project** — predict default risk from Home Credit application data, decide **approve / decline**, explain declines with **SHAP**, serve via **FastAPI** + **Streamlit dashboard**.

| | |
|--|--|
| **Author** | Sudhanshu Kumar|
| **Primary model pack** | `models/full_data/` (**307,511** rows) |
| **Production model** | **XGBoost** (test ROC-AUC **0.769**, accuracy **86.2%**, recall **40.1%**) |
| **Decision policy** | **`middle_ground`** τ=**0.51** (val-only; see [`docs/OPERATING_POLICY.md`](docs/OPERATING_POLICY.md)) |
| **Lifecycle** | Train / val / test + dummy baseline + HP tuning |
| **Stack** | scikit-learn · XGBoost · SHAP · FastAPI · Streamlit |

Live status: [`STATUS.md`](STATUS.md) · Mentor Q&A: [`docs/MENTOR_VIVA_QA.md`](docs/MENTOR_VIVA_QA.md) · Policy: [`docs/OPERATING_POLICY.md`](docs/OPERATING_POLICY.md)

---

## 1. Problem

Home Credit `TARGET = 1` means **payment difficulties** (~8% of applicants).  
We map predicted default probability to a credit-card-style decision:

| Rule | Decision |
|------|----------|
| P(default) ≥ τ | **decline** |
| P(default) < τ | **approve** |

Threshold τ is chosen on the **validation** set with the **`middle_ground`** policy
(accuracy ≥ 80% and recall ≥ 35% on val, then max F1); reported metrics use the **test** set only.
Full write-up: [`docs/OPERATING_POLICY.md`](docs/OPERATING_POLICY.md).

---

## 2. Academic ML lifecycle

```text
Problem → Data → EDA → Features → Stratified train/val/test
  → Pipeline preprocess (fit train only)
  → Dummy baseline + Logistic Regression + XGBoost
  → Select model & threshold on VALIDATION
  → Final metrics on TEST
  → SHAP explanations → API / Dashboard
```

Details: [`docs/ML_LIFECYCLE.md`](docs/ML_LIFECYCLE.md)

---

## 3. Current results (full data + tuning — primary)

**Source:** `models/full_data/metrics.json` · `models/full_data/metrics_summary.md`  
**Split:** train 196,806 · val 49,202 · test 61,503 · base rate ≈ 8.07%  
**HP tuning:** completed (random search on train/val; test never used)  
**τ = 0.510** (`middle_ground` on validation — see [`docs/OPERATING_POLICY.md`](docs/OPERATING_POLICY.md))

### Production operating point (XGBoost, test)

| Accuracy | Bal. Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | Lift@τ |
|----------|----------|-----------|--------|-----|---------|--------|--------|
| **86.20%** | **65.18%** | **26.54%** | **40.10%** | 0.319 | **0.769** | **0.261** | **3.29×** |

### All models @ production τ

| Model | Accuracy | ROC-AUC | Precision@τ | Recall@τ | Lift@τ |
|-------|----------|---------|-------------|----------|--------|
| Dummy (stratified) | 85.3% | 0.502 | 0.084 | 0.083 | 1.04× |
| Logistic Regression | 70.1% | 0.752 | 0.165 | 0.668 | 2.05× |
| **XGBoost (production)** | **86.2%** | **0.769** | **0.265** | **0.401** | **3.29×** |

Best XGB (val): n_estimators=350, max_depth=6, learning_rate=0.05, min_child_weight=5, subsample=0.7, scale_pos_weight≈5.69.

Under ~8% defaults, raw precision is still modest; **lift ≈ 3.3×** and **ROC-AUC 0.77 vs dummy 0.50** show real ranking skill.  
See [`docs/OPERATING_POLICY.md`](docs/OPERATING_POLICY.md) · [`docs/IMBALANCE_AND_METRICS_FAQ.md`](docs/IMBALANCE_AND_METRICS_FAQ.md) · [`STATUS.md`](STATUS.md).

### Other packs (comparison only)

| Pack | Rows | Note |
|------|------|------|
| `models/` | 50k sample | Older LR model (~0.749 AUC) — not default serve |
| `*multitable_30k*` | 30k + history tables | Research ablation (~0.759 LR) |

---

## 4. Repository layout

```text
credit-risk-assessment/
├── STATUS.md                 ← current numbers & checklist
├── README.md
├── requirements.txt
├── docs/                     ← lifecycle, tuning, FAQ, presentation
├── reports/eda/              ← EDA report
├── models/
│   ├── full_data/            ← PRIMARY full-data academic model
│   └── *.joblib / metrics*   ← sample & multitable comparisons
├── src/credit_risk/          ← package (train, predict, explain, api, tuning, eda)
├── dashboard/app.py          ← Streamlit UI
├── scripts/
└── tests/
```

---

## 5. Setup

```bash
cd ~/credit-risk-assessment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Dataset default:

`~/Downloads/home-credit-default-risk/application_train.csv`

---

## 6. Run dashboard (enter values & score)

Uses **full-data** model automatically when present:

```bash
cd ~/credit-risk-assessment
.venv/bin/python -m streamlit run dashboard/app.py
```

Open **http://localhost:8501** → Score applicant → presets or your values → Run assessment.

Force sample model:

```bash
export CREDIT_RISK_MODELS_DIR="\$PWD/models"
```

---

## 7. Run API

```bash
.venv/bin/python -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000
```

Docs: http://127.0.0.1:8000/docs  
Endpoints: `/health` · `/metrics` · `/features` · `/assess` · `/explain`

---

## 8. Train

```bash
# Full academic lifecycle (307k) → models/full_data/
.venv/bin/python scripts/run_academic_lifecycle.py

# Fast iteration sample
.venv/bin/python -m credit_risk.train --sample-size 30000 --model auto --no-eda

# Optional multi-table features
.venv/bin/python -m credit_risk.train --sample-size 30000 --multi-table --model auto

# Re-run full data + hyperparameter tuning (already completed once)
.venv/bin/python -m credit_risk.train --sample-size 0 --tune --tune-iter 10 --artifacts models/full_data/credit_risk_bundle.joblib --metrics models/full_data/metrics.json --no-eda
```

Tuning docs: [`docs/HYPERPARAMETER_TUNING.md`](docs/HYPERPARAMETER_TUNING.md) — **primary pack already includes a completed tune run.**

---

## 9. Tests

```bash
.venv/bin/python -m pytest -q
```

---

## 10. Presentation

- **Word-for-word 5-min script:** [`docs/PRESENTATION_SCRIPT_5MIN.md`](docs/PRESENTATION_SCRIPT_5MIN.md)
- Demo checklist / Q&A: [`docs/PRESENTATION_GUIDE.md`](docs/PRESENTATION_GUIDE.md)

**One-liner for faculty:**

> Full Home Credit train (307k), stratified train/val/test, dummy baseline ROC-AUC ≈ 0.50, production XGBoost test ROC-AUC ≈ 0.77, threshold on validation, SHAP decline reasons, FastAPI + Streamlit.

---

## 11. License & data

MIT (see `LICENSE`). Home Credit data under Kaggle terms — educational use; not a production bank system.
