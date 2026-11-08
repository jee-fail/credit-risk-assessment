# Project status (source of truth)

**Last synced:** middle-ground operating policy (`τ=0.51`, validation-selected)  
**Methodology:** [`docs/OPERATING_POLICY.md`](docs/OPERATING_POLICY.md)  
**Numbers:** [`models/full_data/metrics_summary.md`](models/full_data/metrics_summary.md)  
**Mentor / viva Q&A:** [`docs/MENTOR_VIVA_QA.md`](docs/MENTOR_VIVA_QA.md)

---

## Primary production model

| Field | Value |
|-------|--------|
| **Pack** | `models/full_data/` |
| **Rows** | **307,511** (entire `application_train.csv`) |
| **Lifecycle** | Academic train / validation / test |
| **Hyperparameter tuning** | **Completed** (holdout random search; TEST never used) |
| **Production algorithm** | **XGBoost** |
| **Threshold strategy** | **`middle_ground`** (validation only) |
| **Threshold τ** | **0.5100** |
| **Test Accuracy** | **86.20%** |
| **Test Balanced accuracy** | **65.18%** |
| **Test Precision** | **26.54%** |
| **Test Recall** | **40.10%** |
| **Test F1** | **0.319** |
| **Test ROC-AUC** | **0.7690** (threshold-independent) |
| **Test PR-AUC** | **0.2610** |
| **Lift@τ** | **3.29×** |
| **Dummy baseline ROC-AUC** | **0.5016** |

### Confusion matrix (test, production τ)

| | Predicted non-default | Predicted default |
|--|----------------------|-------------------|
| **Actual non-default** | TN = 51,027 | FP = 5,511 |
| **Actual default** | FN = 2,974 | TP = 1,991 |

Specificity ≈ **90.3%** · NPV ≈ **94.5%** · FPR ≈ **9.7%** · FNR ≈ **59.9%**

### Full test comparison @ production τ

| Model | Accuracy | Bal. Acc | ROC-AUC | Precision | Recall | Lift |
|-------|----------|----------|---------|-----------|--------|------|
| Dummy | 85.28% | 50.16% | 0.502 | 8.4% | 8.3% | 1.04× |
| Logistic Regression | 70.06% | 68.58% | 0.752 | 16.5% | 66.8% | 2.05× |
| **XGBoost** | **86.20%** | **65.18%** | **0.769** | **26.5%** | **40.1%** | **3.29×** |

### Policy comparison (same XGB scores; each τ from validation)

| Policy | τ | Acc | Recall | Note |
|--------|---|-----|--------|------|
| youden | 0.30 | 69.0% | 71.2% | High recall, lower accuracy |
| **middle_ground (prod)** | **0.51** | **86.2%** | **40.1%** | **Accuracy ≥80% & recall ≥35% on val, max F1** |
| max_accuracy | 0.76 | 91.9% | 7.4% | Accuracy-only; recall too low for risk use |

Split: train **196,806** / val **49,202** / test **61,503**.

### Best hyperparameters (from search — unchanged)

| Param | Value |
|-------|--------|
| n_estimators | 350 |
| max_depth | 6 |
| learning_rate | 0.05 |
| min_child_weight | 5 |
| subsample | 0.7 |
| colsample_bytree | 1.0 |
| reg_lambda | 0.5 |
| gamma | 0.0 |
| scale_pos_weight | ≈ 5.69 |

---

## Ethical notes

- Threshold selected on **validation only**; test used once for reporting.
- **No model retrain** for policy change; probabilities unchanged.
- **No fabricated metrics**; constraints fixed before looking at test.
- Full write-up: `docs/OPERATING_POLICY.md`.

---

## Academic lifecycle checklist

```text
[x] Problem definition
[x] Data load (full 307k)
[x] EDA → reports/eda/
[x] Feature engineering
[x] Preprocess Pipeline (fit train only)
[x] Stratified train/val/test
[x] Dummy baseline
[x] Logistic Regression + XGBoost
[x] Hyperparameter tuning (train fit, val select)
[x] Model selection on validation
[x] Threshold on validation (middle_ground policy)
[x] Refit train+val with frozen hyperparams
[x] Final metrics on held-out test
[x] SHAP + API + Dashboard
[x] Operating policy documentation
[x] Docs + tests
```

---

## Commands

```powershell
cd "~/credit-risk-assessment"

# Dashboard / API
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
.\.venv\Scripts\python.exe -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000

# Tests (includes 1000-case spectrum suite)
.\.venv\Scripts\python.exe -m pytest -q

# Spectrum 1000 report only
.\.venv\Scripts\python.exe scripts\run_spectrum_1000.py

# Re-apply middle-ground threshold (no retrain)
.\.venv\Scripts\python.exe scripts\retarget_threshold.py --strategy middle_ground
```

