# Tests

| File | What it verifies |
|------|------------------|
| `test_train.py` | Both models train on real Home Credit sample; ROC-AUC written |
| `test_predict.py` | Decision + probability; threshold rule consistency |
| `test_explain.py` | SHAP top features for a declined case |
| `test_api.py` | FastAPI `/health`, `/assess`, `/explain`, validation errors |
| `test_evaluation.py` | Lift, ranking metrics, middle_ground threshold |
| `test_tuning.py` | Hyperparameter search path |
| `test_spectrum_1000.py` | **1000** spectrum applicants: decision logic + risk ordering |
| `spectrum_generator.py` | Deterministic case factory (risk_level 0→1) |

## Spectrum 1000

Generates applicants from **safe → risky** (EXT_SOURCE, leverage, age, employment, categoricals), scores production XGBoost (and LR), and asserts:

- \(0 \le P(\text{default}) \le 1\)
- `decision == decline` iff \(P \ge \tau\)
- `risk_score == default_probability`
- Message matches decision
- Mean risk and decline rate rise with designed risk level
- Safest pole **approve**, riskiest pole **decline** (full-data pack)

```powershell
# Full report (JSON + markdown)
.\.venv\Scripts\python.exe scripts\run_spectrum_1000.py

# Pytest
.\.venv\Scripts\python.exe -m pytest tests/test_spectrum_1000.py -q
```

Artifacts: `reports/spectrum_1000/`

## Run all tests

```powershell
cd "C:\Users\Ibteshamul Haque\credit-risk-assessment"
.\.venv\Scripts\python.exe -m pytest -q
```
