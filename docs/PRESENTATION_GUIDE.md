# Presentation guide (faculty / viva)

## One-sentence pitch

> “I built an end-to-end credit risk system on the full Home Credit train set (307k applications): stratified train/val/test, dummy baseline, hyperparameter-tuned logistic regression and XGBoost, a validation-selected **middle-ground** decision threshold (accuracy 86%, recall 40%), test ROC-AUC 0.77, SHAP decline reasons, and a FastAPI + Streamlit interface.”

## Numbers to memorize

| Item | Value |
|------|--------|
| Rows | 307,511 |
| Production threshold τ | **0.51** (`middle_ground` on validation) |
| **XGB test accuracy** | **86.2%** |
| **XGB test recall** | **40.1%** |
| **XGB test precision** | **26.5%** |
| XGB balanced accuracy | 65.2% |
| **XGB test ROC-AUC** | **0.769** (tuned) |
| Lift@τ (XGB) | **3.29×** |
| Dummy ROC-AUC | ~0.50 |
| LR test ROC-AUC | 0.75 |
| Base rate | ~8% |
| HP tuning | Yes — train/val only |

Policy details: [`OPERATING_POLICY.md`](OPERATING_POLICY.md)

## Demo order (10 min)

1. **`STATUS.md` / `README.md`** — problem + lifecycle  
2. **`docs/OPERATING_POLICY.md`** — ethical middle-ground threshold  
3. **`models/full_data/metrics_summary.md`** — dummy vs LR vs XGB + policy table  
4. **Dashboard** — high-risk preset → decline + SHAP; low-risk → approve  
5. **API** `/docs` or `pytest -q`  

### Dashboard

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

### API

```powershell
.\.venv\Scripts\python.exe -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000
```

## Likely questions

**Full exhaustive bank:** [`MENTOR_VIVA_QA.md`](MENTOR_VIVA_QA.md) (~90+ questions with answers).

**Why not maximize accuracy alone?** Max-accuracy τ gives ~92% accuracy but only ~7% recall — misses most defaulters. We require recall ≥ 35% on validation while keeping accuracy ≥ 80%.  

**Why is precision ~27%?** Base rate ~8%; among declined applicants we still have **3.3× lift** vs random.  

**Train/val/test?** Models fit on train; model + τ on validation; metrics on test.  

**Full data?** Yes — primary pack is 307k, not only a sample.  

**Hyperparameter tuning?** Yes on full data: random search on train, select on validation, report on test. Best XGB test AUC 0.769.

## Files to open

| Priority | Path |
|----------|------|
| High | `STATUS.md` |
| High | `docs/OPERATING_POLICY.md` |
| High | `docs/MENTOR_VIVA_QA.md` |
| High | `models/full_data/metrics_summary.md` |
| High | `docs/ML_LIFECYCLE.md` |
| High | `dashboard/app.py` (live) |
| Medium | `docs/IMBALANCE_AND_METRICS_FAQ.md` |
