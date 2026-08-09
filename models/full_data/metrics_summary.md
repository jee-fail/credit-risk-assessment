# Model metrics summary (FULL DATA — MIDDLE_GROUND THRESHOLD)

- Production model: **xgboost**
- Rows used: **307511**
- Threshold strategy: **middle_ground** (chosen on **validation** only)
- Threshold τ: **0.5100**
- Final metrics on: **test**
- Test base rate: **8.07%**
- Models retrained for this retarget: **no**
- Methodology: [`docs/OPERATING_POLICY.md`](../../docs/OPERATING_POLICY.md)

## Headline (production operating point — test)

| Metric | XGBoost (test) |
|--------|----------------|
| **Accuracy** | **86.20%** |
| **Balanced accuracy** | **65.18%** |
| **Precision** | **26.54%** |
| **Recall** | **40.10%** |
| F1 | 0.3194 |
| ROC-AUC | 0.7690 |
| PR-AUC | 0.2610 |
| Lift@τ | 3.29x |
| Specificity | ~90.3% |
| NPV | ~94.5% |

## All models @ production τ

| Model | Accuracy | Bal. Acc | ROC-AUC | Precision | Recall | Lift |
|-------|----------|----------|---------|-----------|--------|------|
| Dummy | 0.8528 | 0.5016 | 0.5016 | 0.0837 | 0.0828 | 1.04x |
| Logistic Regression | 0.7006 | 0.6858 | 0.7525 | 0.1652 | 0.6681 | 2.05x |
| **XGBoost** | **0.8620** | **0.6518** | **0.7690** | **0.2654** | **0.4010** | **3.29x** |

## Policy comparison (same model; each τ from validation)

| Policy (val-selected) | τ | Test Acc | Bal Acc | Precision | Recall | F1 | Lift |
|-----------------------|---|----------|---------|-----------|--------|----|------|
| youden | 0.303 | 68.98% | 70.01% | 16.69% | 71.24% | 0.271 | 2.07x |
| f1 | 0.525 | 86.82% | 64.36% | 27.14% | 37.58% | 0.315 | 3.36x |
| fixed | 0.500 | 85.68% | 65.69% | 25.99% | 41.85% | 0.321 | 3.22x |
| middle_ground **(production)** | 0.510 | 86.20% | 65.18% | 26.54% | 40.10% | 0.319 | 3.29x |
| max_accuracy | 0.760 | 91.88% | 53.33% | 48.09% | 7.35% | 0.128 | 5.96x |

Operating point (test): TP=1991 FP=5511 TN=51027 FN=2974
