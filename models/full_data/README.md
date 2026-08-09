# Full-data academic model (PRIMARY — tuned)

Trained on **all** Home Credit `application_train.csv` rows (**307,511**) with **completed hyperparameter tuning**.

| Field | Value |
|-------|--------|
| Lifecycle | `academic_v2_train_val_test` |
| Hyperparameter tuning | **Completed** (holdout; test never used) |
| Train / val / test | 196,806 / 49,202 / 61,503 |
| Production | **xgboost** |
| Threshold strategy | **`middle_ground`** (validation only) |
| Threshold τ | **0.5100** |
| Test accuracy | **86.20%** |
| Test balanced accuracy | **65.18%** |
| Test precision | **26.54%** |
| Test recall | **40.10%** |
| Test ROC-AUC | **0.7690** |
| Test PR-AUC | **0.2610** |
| Lift@τ | **3.29×** |
| Dummy baseline AUC | 0.5016 |

Methodology: [`docs/OPERATING_POLICY.md`](../../docs/OPERATING_POLICY.md)  
Human table: [`metrics_summary.md`](metrics_summary.md)

Best XGB params: n_estimators=350, max_depth=6, learning_rate=0.05, min_child_weight=5, subsample=0.7, colsample_bytree=1.0, reg_lambda=0.5, scale_pos_weight≈5.69.

## Default for serve

`credit_risk.config` loads this pack when the joblib exists.

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
.\.venv\Scripts\python.exe -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000
```

## Re-apply operating policy (no retrain)

```powershell
.\.venv\Scripts\python.exe scripts\retarget_threshold.py --strategy middle_ground
```

## Retrain with tuning + middle-ground threshold

```powershell
.\.venv\Scripts\python.exe -m credit_risk.train --sample-size 0 --tune --tune-iter 10 --threshold-strategy middle_ground --artifacts models/full_data/credit_risk_bundle.joblib --metrics models/full_data/metrics.json --no-eda
```
