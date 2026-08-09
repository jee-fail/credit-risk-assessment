# Project basics cheat-sheet (viva / mentor)

Quick answers for common “basic” ML questions.  
**Source of truth:** `STATUS.md`, `models/full_data/metrics.json`

---

## 1. Problem

| Question | Answer |
|----------|--------|
| **What is the problem?** | Binary classification: predict **payment difficulties** (default risk) |
| **Target column** | `TARGET` (1 = difficulty, 0 = no difficulty) |
| **Positive class** | Default / payment difficulty (~**8.07%**) |
| **Decision** | Approve if \(P(\text{default}) < \tau\), else **decline** |
| **Dataset** | Home Credit `application_train.csv` |
| **Rows used (primary)** | **307,511** (full train file) |
| **Primary features** | Application table + derived (not multi-table in production) |

---

## 2. Models used

| Model | Role | Library |
|-------|------|---------|
| **DummyClassifier** (stratified) | Null baseline | scikit-learn |
| **Logistic Regression** | Linear baseline / challenger | scikit-learn |
| **XGBoost** (`XGBClassifier`) | **Production model** | xgboost |

**Not used:** Neural nets, epochs, batch size, CNN/RNN, SVM, Random Forest (as production).

### Why no “number of epochs”?
This is **not deep learning**.  
- **XGBoost:** boosting rounds ≈ **`n_estimators` (trees)** = **350** (tuned)  
- **Logistic Regression:** optimizer steps ≈ **`max_iter`** = **500** (tuned)  
There is **no epoch** in the neural-net sense.

---

## 3. Train / validation / test split

| Item | Value |
|------|--------|
| Method | **Stratified** `train_test_split` |
| Random seed | **`RANDOM_STATE = 42`** |
| Test size | **20%** of all data |
| Val size | **20%** of remaining (train+val pool) |
| Train rows | **196,806** |
| Validation rows | **49,202** |
| Test rows | **61,503** |
| Default rate (each split) | ~**8.07%** |

**Who uses what**

| Split | Used for |
|-------|----------|
| Train | Fit models, HP candidates |
| Validation | Model choice, HP selection, **threshold τ** |
| Test | **Final metrics only** (never for tuning) |

After selection: optional **refit on train+val**, then evaluate on test.

---

## 4. Preprocessing techniques

All inside a **scikit-learn `Pipeline` + `ColumnTransformer`** (fit on train only → no leakage).

| Step | Numeric | Categorical |
|------|---------|-------------|
| **Missing values** | **Median** imputation | **Most frequent** imputation |
| **Scaling** | **StandardScaler** (for **LR only**) | — |
| **Encoding** | — | **One-hot** (`OneHotEncoder`, `handle_unknown="ignore"`) |
| **XGBoost scaling** | **No** StandardScaler (trees don’t need it) | same OHE path |

### Feature engineering (before preprocess)
- Domain **ratios:** credit/income, annuity/income, annuity/credit, credit/goods, income per person, etc.
- **EXT_SOURCE** aggregates: mean, min, max, prod, std, missing count  
- **DAYS_EMPLOYED** clean: sentinel `365243` → anomaly flag + NaN + years employed  
- Production feature count after FE: **140** (**124** numeric, **16** categorical raw-type before OHE expands cats)

### Multi-table?
- Research ablation exists (`--multi-table`: bureau, previous apps, installments, credit card, POS)  
- **Production pack = application-only + derived** (`multi_table: false`)

---

## 5. Class imbalance handling

| Technique | Where |
|-----------|--------|
| Stratified splits | All splits |
| **`class_weight="balanced"`** | Logistic Regression |
| **`scale_pos_weight ≈ n_neg/n_pos ≈ 5.69`** | XGBoost (tuned) |
| Threshold policy | `middle_ground` (not SMOTE in production) |
| Metrics | ROC-AUC, PR-AUC, lift, precision@top-k — not accuracy alone |

**SMOTE:** not used in primary pipeline (optional research; weights + threshold preferred).

---

## 6. Hyperparameter tuning

| Item | Value |
|------|--------|
| Done? | **Yes** (full data) |
| Method | **Holdout random search** (fit train, score **validation**) |
| Test used in search? | **No** |
| LR candidates | **5** |
| XGB candidates | **10** |
| Selection metric (search) | Validation **ROC-AUC** (also tracks PR-AUC) |

### Best Logistic Regression (tuned)

| Param | Value |
|-------|--------|
| C | **0.5** |
| solver | **saga** (from search; code default builder uses lbfgs) |
| max_iter | **500** |
| class_weight | balanced |

### Best XGBoost (tuned) — **production**

| Param | Value |
|-------|--------|
| n_estimators | **350** |
| max_depth | **6** |
| learning_rate | **0.05** |
| min_child_weight | **5** |
| subsample | **0.7** |
| colsample_bytree | **1.0** |
| reg_lambda | **0.5** |
| gamma | **0.0** |
| scale_pos_weight | **≈ 5.69** |
| eval_metric | auc |
| tree_method | hist (in default builder) |

**Learning rate** here is **boosting step size**, not neural-net LR schedule.

---

## 7. Decision threshold (operating policy)

| Item | Value |
|------|--------|
| Production strategy | **`middle_ground`** |
| τ | **0.5100** |
| Selected on | **Validation only** |
| Rule | Decline if \(P(\text{default}) \ge 0.51\) |
| How chosen | Val accuracy ≥ **80%**, recall ≥ **35%**, then max **F1** |

| Alternate policies (for discussion) | Approx test acc | Approx recall |
|-------------------------------------|-----------------|---------------|
| Youden | ~69% | ~71% |
| middle_ground (prod) | **86%** | **40%** |
| max_accuracy | ~92% | ~7% |

---

## 8. Evaluation metrics (test, XGBoost @ τ=0.51)

| Metric | Value | One-line meaning |
|--------|-------|------------------|
| **Accuracy** | **86.20%** | Correct approve/decline rate |
| **Balanced accuracy** | **65.18%** | Mean of TPR & TNR |
| **Precision** | **26.54%** | Of declines, true defaults |
| **Recall (sensitivity)** | **40.10%** | Of true defaults, caught |
| **Specificity** | **~90.3%** | Of non-defaults, correctly approved-side |
| **NPV** | **~94.5%** | Of approves, truly non-default |
| **F1** | **0.319** | Harmonic mean P & R |
| **ROC-AUC** | **0.769** | Ranking quality (τ-independent) |
| **PR-AUC** | **0.261** | Ranking under imbalance |
| **Lift@τ** | **3.29×** | Precision / base rate |
| **Dummy ROC-AUC** | **~0.50** | Chance baseline |

### Confusion matrix (test)

|  | Pred non-default | Pred default |
|--|------------------|--------------|
| Actual non-default | TN **51,027** | FP **5,511** |
| Actual default | FN **2,974** | TP **1,991** |

---

## 9. Explainability

| Item | Answer |
|------|--------|
| Method | **SHAP** |
| XGBoost | TreeExplainer |
| Logistic Regression | LinearExplainer / coefficients |
| Output | Top-k features raising/lowering risk |
| Dashboard default top_k | **8** |

---

## 10. Software / serving

| Piece | Tech |
|-------|------|
| Language | Python 3.10+ (dev used 3.13) |
| Core ML | pandas, numpy, scikit-learn, xgboost, shap |
| API | FastAPI + uvicorn |
| UI | Streamlit + Plotly |
| Artifacts | joblib bundle `credit_risk_bundle.joblib` |
| Config seed | 42 |
| Tests | pytest (API, train sample, predict, explain, **1000 spectrum cases**) |

---

## 11. What we did **not** use (honest)

| Topic | Status |
|-------|--------|
| Epochs / mini-batches / GPU NN | **No** |
| SMOTE / ADASYN (production) | **No** |
| PCA / feature selection filter | **Not primary** |
| K-fold nested CV as main protocol | Holdout train/val/test primary; CV optional in tuner |
| Probability calibration (Platt/isotonic) | **Not in primary deliverable** |
| Full fair-lending audit | **No** (limitation) |
| Production multi-table full 307k | Research pack only |

---

## 12. Libraries / stack one-liner

> “Tabular credit risk pipeline in Python: **pandas** prep, **sklearn Pipeline** preprocess, **Logistic Regression + XGBoost** with imbalance weights, **holdout HP search**, **middle-ground threshold**, **SHAP** explanations, served via **FastAPI** and **Streamlit**.”

---

## 13. 30-second oral version

> Dataset Home Credit, 307k rows, 8% defaults. Stratified train/val/test. Median/mode impute, one-hot, scale for LR. Models: dummy, logistic regression, XGBoost production. Tuned XGB: 350 trees, depth 6, lr 0.05, scale_pos_weight ~5.7. No neural epochs. Threshold 0.51 middle-ground on validation. Test: accuracy 86%, recall 40%, ROC-AUC 0.77, lift 3.3×. SHAP for explanations.

---

## 14. File map

| Need | Open |
|------|------|
| Numbers | `STATUS.md` |
| Metrics table | `models/full_data/metrics_summary.md` |
| Threshold ethics | `docs/OPERATING_POLICY.md` |
| Long Q&A | `docs/MENTOR_VIVA_QA.md` |
| This sheet | `docs/BASICS_CHEATSHEET.md` |
