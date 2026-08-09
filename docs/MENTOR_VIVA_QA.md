# Mentor / viva Q&A — exhaustive preparation

**Project:** Credit Risk Assessment System  
**Author:** Ibteshamul Haque  
**Primary pack:** `models/full_data/` (307,511 rows, XGBoost, τ=`middle_ground` = **0.51**)  
**Live numbers:** `STATUS.md` · `models/full_data/metrics_summary.md` · `docs/OPERATING_POLICY.md`

Use this as a **script book**. Short answers first; longer technical depth if they dig.

---

## 0. One-minute elevator (if they open with “what did you do?”)

> I built an end-to-end credit default risk system on the full Home Credit application train set (**307,511** rows). I ran stratified **train/validation/test**, a **dummy baseline**, **logistic regression** and **XGBoost** with imbalance handling and hyperparameter tuning, selected an ethical **middle-ground decision threshold** on validation only (**accuracy 86.2%**, **recall 40.1%**, **ROC-AUC 0.769** on test), added **SHAP** explanations for declines, and shipped a **FastAPI + Streamlit** interface. Everything is documented and tested.

---

## 1. Problem definition & domain

### Q1. What problem are you solving?
**A:** Binary classification of **payment difficulties** (Home Credit `TARGET=1`) from application features, mapped to **approve / decline** for credit. Not a bank production system—an academic ML lifecycle with explainability and serving.

### Q2. Who is the user / stakeholder?
**A:** Academic reviewers and demo users (loan officer style). The dashboard lets a user enter applicant features, get **P(default)**, a decision, and **SHAP** reasons if declined.

### Q3. What is the business decision rule?
**A:**  
- If \(P(\text{default}) \ge \tau\) → **decline**  
- Else → **approve**  
Production \(\tau = 0.51\) via `middle_ground` policy (see §7).

### Q4. Why is this hard?
**A:** Strong **class imbalance** (~**8.07%** defaults), missing data, mixed numeric/categorical features, and a **cost asymmetry** (missing a defaulter vs declining a good applicant). Ranking quality and threshold policy both matter.

### Q5. Is this the Kaggle competition score?
**A:** We use the same **dataset family** and the competition’s primary skill metric is **ROC-AUC**. We are not claiming a leaderboard submission. Application-only models typically land ~**0.70–0.76** AUC; we are at **0.769** with tuning.

---

## 2. Dataset & EDA

### Q6. What data did you use?
**A:** Home Credit `application_train.csv` — **307,511** applications, **TARGET** = payment difficulties. Optional multi-table research on bureau / previous / installments / credit card / POS (ablation pack, not production default).

### Q7. What is the class distribution?
**A:** TARGET=1 ≈ **8.07%** (24,825 positives; 282,686 negatives). Neg/pos ratio ≈ **11.4**. Documented in `reports/eda/eda_report.md`.

### Q8. What did EDA show?
**A:** Heavy imbalance; high missingness on housing-related fields (e.g. COMMONAREA ~70%); mix of ~104 numeric and ~16 categorical application fields before derived features. Implications: stratified splits, imputation inside Pipeline, imbalance-aware metrics and training.

### Q9. Did you drop highly missing columns?
**A:** No global drop of high-missing features in production. We use **median/mode imputation** inside sklearn Pipeline so inference and training stay consistent. Trees handle missingness patterns reasonably; LR gets imputed+scaled numerics.

### Q10. Any data quality traps?
**A:** Yes—e.g. `DAYS_EMPLOYED = 365243` is a known sentinel for “unemployed / anomaly.” We flag `DAYS_EMPLOYED_ANOMALY`, set clean employment days to NaN, and derive `YEARS_EMPLOYED`.

### Q11. Train on application only—why not all tables in production?
**A:** Full multi-table on 307k is heavier and was researched as a **30k multi-table ablation**. Production pack is **application + derived ratios** (140 features) for a clean academic primary result. Multi-table is the main roadmap for higher AUC (documented in research notes).

---

## 3. Features & preprocessing

### Q12. How many features?
**A:** Production: **140** after application-level derived engineering (ratios, EXT_SOURCE aggregates, employment clean-up). Multi-table research pack expands further (~300+).

### Q13. What derived features did you create?
**A (examples):**  
`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `ANNUITY_CREDIT_RATIO`, `CREDIT_GOODS_RATIO`, `INCOME_PER_PERSON`, `EXT_SOURCE_MEAN/MIN/MAX/PROD/STD`, employment anomaly cleaning. These mirror strong public Home Credit baselines.

### Q14. Preprocessing pipeline?
**A:** sklearn `Pipeline` + `ColumnTransformer`:  
- Numeric: **median impute** (+ **StandardScaler** for LR)  
- Categorical: **most_frequent impute** + **OneHotEncoder(handle_unknown="ignore")**  
Fitted on **train only** (or train+val after HP freeze for final refit)—never on test.

### Q15. Why impute inside the Pipeline?
**A:** Prevents **leakage** (statistics computed only on fit fold) and guarantees the **same transforms** at API/dashboard inference.

### Q16. Feature scaling for XGBoost?
**A:** Trees don’t need scaling; LR does. XGB preprocessor uses impute without StandardScaler; LR includes scaling.

### Q17. Target leakage risk?
**A:** We do **not** use future payment outcomes or post-default fields as features—only application-time (and historical bureau tables when multi-table). `TARGET` is the label only. IDs (`SK_ID_CURR`) are not used as predictive features.

### Q18. Multicollinearity?
**A:** LR can suffer; we rely on regularization (`C` tuned). Trees are more robust. We don’t claim VIF cleanup as a formal step—honest limitation if asked.

---

## 4. Train / validation / test protocol

### Q19. How did you split data?
**A:** Stratified:  
1. **20% test** held out first  
2. Remaining → **80% train / 20% val** of remaining  
Sizes: train **196,806** / val **49,202** / test **61,503**  
Seed: `RANDOM_STATE = 42`.

### Q20. What is each split for?
| Split | Role |
|-------|------|
| Train | Fit models / HP candidates |
| Validation | Model choice, HP selection, **threshold** |
| Test | **Final metrics only** (once per frozen setup) |

### Q21. Did you look at the test set during development?
**A:** Lifecycle is designed so model + τ selection is **validation-only**. Retargeting threshold uses the same protocol. We do not optimize τ on test.

### Q22. Did you refit on train+val?
**A:** Yes (default): after freezing HP and model choice from val, **refit** selected pipelines on train+val, then evaluate on test. Threshold remains the one chosen on validation scores (policy fixed before final test report).

### Q23. Why stratified splits?
**A:** Preserve ~8% default rate in each fold so metrics are comparable and rare positives aren’t missing from a fold by chance.

---

## 5. Models

### Q24. Which models and why?
**A:**  
1. **Dummy (stratified)** — null baseline (~ROC-AUC 0.50)  
2. **Logistic regression** — strong linear baseline, interpretable coefficients path  
3. **XGBoost** — nonlinear interactions, strong tabular performance; **production**

### Q25. Why not only neural nets / deep learning?
**A:** Tabular credit data often favors GBMs; NNs need more tuning/data prep and are harder to explain. Scope is classical + GBM academic stack with SHAP.

### Q26. Why not only accuracy-maximizing classifier?
**A:** Always predicting “no default” gets ~**92% accuracy** and **0% recall**—useless for risk. We use ranking metrics + a constrained threshold policy.

### Q27. How do you handle class imbalance in training?
**A:**  
- LR: `class_weight="balanced"`  
- XGB: `scale_pos_weight ≈ n_neg/n_pos` (tuned; best ≈ **5.69**)  
- Stratified splits  
- Threshold policy (not SMOTE in production)

### Q28. Why not SMOTE?
**A:** Mixed evidence; easy to leak if applied wrong; tree models with scale_pos_weight often work well. We prioritize **correct CV protocol + class weights + threshold**. SMOTE is optional future work, never on test.

### Q29. Production model selection rule?
**A:** Choose higher **validation ROC-AUC** between LR and XGB (auto). Production is **XGBoost** (val AUC ~0.764 during search; test **0.769**).

### Q30. Best XGBoost hyperparameters?
**A (from search):** n_estimators=350, max_depth=6, learning_rate=0.05, min_child_weight=5, subsample=0.7, colsample_bytree=1.0, reg_lambda=0.5, gamma=0.0, scale_pos_weight≈5.69.

### Q31. How was HP tuning done?
**A:** Holdout random search: fit on train, score on **validation**, **test never used**. Completed for full data. Details: `docs/HYPERPARAMETER_TUNING.md`, `metrics.json → hyperparameter_tuning`.

---

## 6. Metrics (expect heavy fire here)

### Q32. What is your accuracy?
**A:** **86.20%** on test at production τ=0.51.

### Q33. Is 86% good?
**A:** Better than a random stratified dummy at the same τ in our table context; **must not** be compared only to 100%. Majority-class “always approve” is ~**91.9%** accuracy with **no risk detection**. Our accuracy is **high while still catching 40% of defaulters**.

### Q34. Why isn’t accuracy 95%+?
**A:** Catching defaulters requires declining some non-defaulters (false positives). Pushing accuracy to 92% (max_accuracy τ=0.76) **destroys recall (~7%)**. We explicitly rejected that as a production policy.

### Q35. What is recall?
**A:** Of true defaulters, fraction we decline. **40.10%** → we catch ~**40 of 100** defaulters; miss ~60 (FN).

### Q36. Is 40% recall low?
**A:** It is a **deliberate tradeoff** under accuracy floors. Youden τ gets ~**71%** recall but only ~**69%** accuracy. For a mentor who wants high accuracy **and** non-trivial risk catch, 40% is the middle ground. In a bank, recall target depends on loss appetite.

### Q37. What is precision?
**A:** Of those we decline, fraction who truly default: **26.54%**.

### Q38. Precision only 27%—is the model bad?
**A:** No. Base rate is 8%. **Lift = 26.5 / 8.07 ≈ 3.3×** → declined group defaults **3.3 times** more often than average. Random screening would be ~8% precision.

### Q39. What is F1?
**A:** Harmonic mean of precision and recall: **0.319**. Used as objective **among feasible** middle_ground thresholds on validation.

### Q40. What is balanced accuracy?
**A:** Mean of sensitivity and specificity: **65.18%**. Better than raw accuracy when classes are imbalanced.

### Q41. What is ROC-AUC?
**A:** Probability a random defaulter scores higher than a random non-defaulter: **0.769** (dummy ~0.50). **Independent of τ.** Primary ranking skill metric for Home Credit-style problems.

### Q42. What is PR-AUC?
**A:** Area under precision–recall curve: **0.261** vs base rate ~0.08 if random. Better reflects positive-class ranking under imbalance.

### Q43. What is lift?
**A:** Precision / base rate. **3.29×** at production τ.

### Q44. Precision@top 10%?
**A:** Among the 10% highest scores, fraction who default ≈ **28.4%** (from ranking metrics; threshold-independent style evaluation).

### Q45. Confusion matrix at production τ?
| | Pred 0 | Pred 1 |
|--|--------|--------|
| Actual 0 | TN **51,027** | FP **5,511** |
| Actual 1 | FN **2,974** | TP **1,991** |

- Specificity ≈ **TN/(TN+FP) ≈ 90.3%**  
- NPV ≈ **TN/(TN+FN) ≈ 94.5%** (approved applicants are mostly non-default)  
- FPR ≈ **9.7%**, FNR ≈ **59.9%**

### Q46. Which metric is “primary”?
**A:** Honest answer:  
- **Ranking skill:** ROC-AUC / PR-AUC  
- **Operating point (mentor-facing):** accuracy + recall + precision together under documented policy  
We **do not** optimize only accuracy.

### Q47. Why report so many metrics?
**A:** Single metrics lie under imbalance. A full suite prevents gaming and answers different stakeholder questions (risk officer vs ops vs academic).

---

## 7. Threshold / operating policy (critical)

### Q48. How did you choose τ=0.51?
**A:** Strategy **`middle_ground`** on **validation** only:  
1. Scan τ from 0.05 to 0.95  
2. Keep candidates with **val accuracy ≥ 80%** and **val recall ≥ 35%**  
3. Maximize **F1** among them  
4. Tie-break: balanced accuracy → accuracy → recall  
Documented in `docs/OPERATING_POLICY.md`.

### Q49. Why not Youden?
**A:** Youden maximizes TPR−FPR → high recall (~71%) but accuracy ~69%. Rejected for reviews that demand higher accuracy.

### Q50. Why not max accuracy?
**A:** τ≈0.76 → accuracy ~**91.9%** but recall ~**7%**—almost never catches defaulters. Unethical as a “risk model” for production policy.

### Q51. Did changing τ retrain the model?
**A:** **No.** Same probabilities; only the cut-off changes. ROC-AUC stays **0.769**.

### Q52. Did you pick τ on the test set?
**A:** **No.** Validation only. Test evaluated after freeze.

### Q53. Can I change τ in the dashboard?
**A:** Yes—override slider for what-if analysis; default is production bundle τ.

### Q54. Policy comparison (memorize)?
| Policy | τ | Acc | Recall |
|--------|---|-----|--------|
| youden | 0.30 | 69% | 71% |
| **middle_ground** | **0.51** | **86%** | **40%** |
| max_accuracy | 0.76 | 92% | 7% |

---

## 8. Results & comparisons

### Q55. Final headline numbers (test, XGB)?
| Metric | Value |
|--------|-------|
| Accuracy | **86.20%** |
| Balanced acc | **65.18%** |
| Precision | **26.54%** |
| Recall | **40.10%** |
| F1 | 0.319 |
| ROC-AUC | **0.769** |
| PR-AUC | **0.261** |
| Lift | **3.29×** |

### Q56. Dummy vs LR vs XGB?
At production τ: XGB best accuracy (**86.2%**) and ROC-AUC (**0.769**). LR has higher recall at this shared τ but lower accuracy (70%). Dummy ~chance ranking (AUC 0.50).

### Q57. Does XGB beat dummy?
**A:** Yes—ROC-AUC 0.769 vs 0.50; lift 3.3× vs ~1.0×; formal `beats_baseline` flags in metrics.json.

### Q58. How does this compare to literature / Kaggle?
**A:** Application-only AUC ~0.75–0.77 is **credible**. Top public solutions often ~0.80 with heavy multi-table FE. We are in the right band for our feature scope.

### Q59. Sample 50k vs full 307k?
**A:** Full data is primary. Sample pack remains for comparison only—not default serve.

---

## 9. Explainability

### Q60. How do you explain declines?
**A:** **SHAP** local explanations: TreeExplainer for XGB; LinearExplainer / coefficient path for LR. Top features by |SHAP| with direction (increased/decreased risk).

### Q61. Why SHAP?
**A:** Consistent local feature attributions; widely accepted for GBM credit models; supports “right to explanation” style demos.

### Q62. Are SHAP values global feature importance?
**A:** Local by default (per applicant). Global importance can be aggregated but is not the primary dashboard story.

### Q63. Can explanations be wrong?
**A:** SHAP is model-faithful, not causal truth. Correlation ≠ causation; correlated features can share credit. We state educational use.

---

## 10. System / software engineering

### Q64. Architecture?
**A:**  
`src/credit_risk/` package (data, EDA, preprocess, train, tune, evaluate, predict, explain, api)  
`dashboard/app.py` Streamlit  
`models/full_data/` artifacts  
`tests/` pytest  
`docs/` methodology  

### Q65. How is the model served?
**A:** joblib bundle with pipelines + threshold + metadata. FastAPI endpoints for assess/explain; Streamlit for interactive UI.

### Q66. How do you avoid train/serve skew?
**A:** Same `Pipeline` object used at train and inference; `prepare_input_frame` aligns columns; derived features recomputed the same way.

### Q67. Testing?
**A:** pytest covers API, evaluation math, explain path, predict, train lifecycle (sample), tuning. Run: `pytest -q` (14+ tests).

### Q68. Reproducibility?
**A:** Fixed seeds, pinned-ish deps in requirements, documented commands, metrics JSON lifecycle log.

### Q69. Configuration?
**A:** Paths via `credit_risk.config` and env vars (`CREDIT_RISK_MODELS_DIR`, data dir, etc.).

---

## 11. Ethics, fairness, compliance

### Q70. Is this ready for a real bank?
**A:** **No.** Educational demo. Real underwriting needs regulatory capital models, fair-lending review, monitoring, champion/challenger, data lineage, and approved features.

### Q71. Fairness / protected attributes?
**A:** Home Credit includes demographics; we did **not** run a full disparate-impact audit (limitation). Production systems must analyze by protected classes and may restrict features by law/policy.

### Q72. Bias risk?
**A:** Historical labels can encode past discrimination; proxy variables exist. Mitigation would include fairness metrics, feature governance, human review—future work.

### Q73. Privacy?
**A:** Use public competition-style data locally; no claim of processing real customer PII in production.

### Q74. Did you fabricate metrics to impress?
**A:** **No.** All metrics from held-out test labels. Policy comparison published including policies that look “worse” on some axes.

---

## 12. Limitations (say these before they attack)

### Q75. What are the main limitations?
1. Application-table-only production features (no full multi-table in primary pack)  
2. Threshold is a **policy**, not a unique mathematical optimum  
3. No probability **calibration** study (Platt/isotonic) in primary deliverable  
4. No full **fairness** audit  
5. No time-based split (competition data isn’t a clean production time series API)  
6. SHAP is correlational  
7. Cost of FP vs FN not dollarized  

### Q76. Biggest improvement if you had more time?
**A:** Full multi-table FE on full data + LightGBM/XGB comparison + calibration + fairness report + cost-sensitive τ.

---

## 13. “Gotcha” / hostile questions

### Q77. “Always approve gets 92% accuracy—your model is worse.”
**A:** Always-approve has **0% recall** and no risk ranking. Ours has **86% accuracy and 40% recall**, ROC-AUC **0.77**. Accuracy alone is the wrong objective for rare defaults.

### Q78. “Why not report 92% accuracy then?”
**A:** We **can** (max_accuracy policy)—but that is not a responsible risk policy. We publish it in the comparison table and **reject** it for production.

### Q79. “You changed threshold after training—is that cheating?”
**A:** No—standard ML practice. Model learns scores; threshold is a decision layer. Cheating would be choosing τ on **test** or fabricating labels. We select on **validation**.

### Q80. “Why are 80% and 35% floors not overfit?”
**A:** They are **a priori policy constraints** for a balanced review, not maximized on test. Test is only for reporting. Alternative policies are fully disclosed.

### Q81. “F1 is still low (~0.32).”
**A:** Under 8% base rate, F1 is naturally modest. Lift and ROC-AUC show skill. F1 was the **selector among feasible high-accuracy thresholds**, not the sole business KPI.

### Q82. “Did data leakage inflate AUC?”
**A:** Pipeline fits on train; test untouched; no target-derived features. If anything, application-only **limits** AUC vs multi-table leakage-free history features.

### Q83. “Why is LR recall higher than XGB at same τ?”
**A:** Same τ on differently calibrated probability scales. LR with class_weight can push more mass above 0.51. Model choice used **validation ROC-AUC**, not recall at a shared τ.

### Q84. “Calibrated probabilities?”
**A:** We use raw `predict_proba`. For pure ranking, calibration is secondary; for threshold interpretability as true risk, calibration is future work.

### Q85. “Cross-validation?”
**A:** Primary protocol is **holdout train/val/test** for clear academic storytelling and tuning. Optional CV mode exists in tuning flags; full nested CV is costlier on 307k.

### Q86. “External validity?”
**A:** Home Credit is specific geography/product mix. Transfer to another portfolio needs retrain and monitoring—standard caveat.

### Q87. “Overfitting?”
**A:** Test AUC ≈ val AUC (~0.76–0.77); dummy baseline; regularization and subsample; no test-driven HP. Overfit would show large val≫test gap.

### Q88. “Why 0.51 not exactly 0.5?”
**A:** Grid search on validation under constraints picked **0.51** as better F1 among feasible points. Close to classic 0.5; not hand-waved.

### Q89. “Can I see the code path?”
**A:** `train.py` lifecycle → `select_threshold(middle_ground)` → artifacts → `predict.assess_applicant` → dashboard/API.

### Q90. “What if mentor demands ≥90% accuracy?”
**A:** Show policy table: ≥90% forces τ into max_accuracy region and **collapses recall**. Offer dual reporting: risk policy vs accuracy policy—same model, honest tradeoff. Do not fake 90% with 40% recall.

---

## 14. Demo walkthrough questions

### Q91. Walk me through the dashboard.
1. Sidebar: model pack, τ  
2. Preset high-risk → Run → decline, high P(default), SHAP bars  
3. Low-risk preset → approve  
4. Metrics tab: accuracy **86.2%**, recall **40%**, ROC-AUC **0.77**, strategy middle_ground  

### Q92. What if SHAP fails?
**A:** Explain module degrades gracefully where possible; tests cover explain path. TreeExplainer is primary for XGB.

### Q93. API endpoints?
**A:** FastAPI assess + explain style routes (see `credit_risk.api` / OpenAPI `/docs` when server runs).

---

## 15. Academic / syllabus alignment

### Q94. Map to ML lifecycle stages.
Problem → Data → EDA → Features → Split → Train → Validate → Threshold → Test → Explain → Deploy (API/UI) → Document. Checklist in `STATUS.md`.

### Q95. What is novel / your contribution?
**A:** Not a new algorithm—**complete rigorous pipeline**: full-data tuning, imbalance-aware evaluation, ethical dual-aware threshold policy, SHAP, serving, tests, and transparent documentation.

### Q96. Research questions you answered?
1. Does XGB beat LR/dummy on application features? **Yes (AUC).**  
2. How does τ policy move accuracy vs recall? **Quantified table.**  
3. Do multi-table features help (ablation)? **Modest gains in 30k study.**  

---

## 16. Numbers cheat-sheet (print this page)

```
Rows:            307,511
Base rate:       8.07%
Split:           196,806 / 49,202 / 61,503
Model:           XGBoost (tuned)
τ:               0.51 (middle_ground, val-only)
Accuracy:        86.20%
Balanced acc:    65.18%
Precision:       26.54%
Recall:          40.10%
F1:              0.319
ROC-AUC:         0.769
PR-AUC:          0.261
Lift:            3.29×
TP/FP/TN/FN:     1991 / 5511 / 51027 / 2974
Dummy AUC:       ~0.50
LR AUC:          0.752
```

---

## 17. If you freeze—safe sentences

1. “That metric is threshold-dependent; ROC-AUC is not.”  
2. “We chose τ on validation under documented constraints; test is report-only.”  
3. “Under 8% defaults, precision must be read with lift.”  
4. “Always-approve accuracy is ~92% and useless for risk.”  
5. “Limitation: application-only features; multi-table is the main upside.”  
6. “This is educational, not a licensed credit engine.”  

---

## 18. Files to open if asked for evidence

| Ask | Open |
|-----|------|
| Numbers | `models/full_data/metrics_summary.md` |
| Status | `STATUS.md` |
| Threshold ethics | `docs/OPERATING_POLICY.md` |
| Lifecycle | `docs/ML_LIFECYCLE.md` |
| Imbalance | `docs/IMBALANCE_AND_METRICS_FAQ.md` |
| Model card | `docs/MODEL_CARD.md` |
| This Q&A | `docs/MENTOR_VIVA_QA.md` |
| EDA | `reports/eda/eda_report.md` |
| Code train | `src/credit_risk/train.py` |
| Live demo | Dashboard http://127.0.0.1:8501 |
