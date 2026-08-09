# Model card — Credit Risk Assessment

## Primary model (production)

| Field | Value |
|-------|--------|
| **Path** | `models/full_data/credit_risk_bundle.joblib` |
| **Lifecycle** | `academic_v2_train_val_test` + hyperparameter tuning |
| **Production algorithm** | **XGBoost** |
| **Rows** | **307,511** |
| **Threshold strategy** | **`middle_ground`** (validation only) |
| **Threshold τ** | **0.510** |
| **Test accuracy** | **86.20%** |
| **Test balanced accuracy** | **65.18%** |
| **Test precision** | **26.54%** |
| **Test recall** | **40.10%** |
| **Test ROC-AUC** | **0.769** |
| **Test PR-AUC** | **0.261** |
| **Lift@τ** | **3.29×** |
| **Intended use** | Educational credit risk approve/decline + SHAP |
| **Operating policy** | [`OPERATING_POLICY.md`](OPERATING_POLICY.md) |

## Training data

Home Credit `application_train.csv`, TARGET = payment difficulties (~8.07% positive).  
140 features (application + derived).

## Split protocol

| Split | Size | Role |
|-------|------|------|
| Train | 196,806 | Fit + HP search candidates |
| Validation | 49,202 | HP selection, model choice, **threshold** |
| Test | 61,503 | Final metrics only |

## Hyperparameter tuning

- Mode: holdout random search (`--tune --tune-mode holdout --tune-iter 10`)
- Fit on train; score on validation; **test never used**
- Best XGB val ROC-AUC: **0.7637** → test **0.7690**
- Best XGB params: n_estimators=350, max_depth=6, lr=0.05, min_child_weight=5, subsample=0.7, colsample_bytree=1.0, reg_lambda=0.5, scale_pos_weight≈5.69

## Test metrics @ production τ = 0.51

| Model | Accuracy | Bal. Acc | ROC-AUC | Precision | Recall | Lift@τ |
|-------|----------|----------|---------|-----------|--------|--------|
| Dummy | 85.3% | 50.2% | 0.502 | 0.084 | 0.083 | 1.04× |
| Logistic Regression | 70.1% | 68.6% | 0.752 | 0.165 | 0.668 | 2.05× |
| **XGBoost** | **86.2%** | **65.2%** | **0.769** | **0.265** | **0.401** | **3.29×** |

Ranking-only (threshold-independent): ROC-AUC / PR-AUC / precision@top-k% unchanged by τ.

## Decision policy

Decline if \(P(\text{default}) \ge \tau\).

**Middle ground selection (validation):** accuracy ≥ 80% and recall ≥ 35%, then maximize F1.  
See [`OPERATING_POLICY.md`](OPERATING_POLICY.md).

## Explainability

SHAP TreeExplainer (XGB) / LinearExplainer (LR).

## Reproducibility

```powershell
# Re-apply threshold policy only
.\.venv\Scripts\python.exe scripts\retarget_threshold.py --strategy middle_ground

# Full retrain
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 0 --tune --tune-iter 10 --threshold-strategy middle_ground --artifacts models/full_data/credit_risk_bundle.joblib --metrics models/full_data/metrics.json
.\.venv\Scripts\python.exe -m pytest -q
```

Seed: `RANDOM_STATE = 42`.
