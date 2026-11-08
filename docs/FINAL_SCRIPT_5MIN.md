# FINAL SCRIPT — Credit Risk Assessment System  
### ~5 minutes · impressive · basic → advanced · all verified numbers

**Duration:** 4:45–5:15  
**Open before you start:** Dashboard at http://localhost:8501  
**Sidebar:** Score with = **Production (xgboost)** · Threshold τ ≈ **0.51**  
**Do not click Run until the live demo section.**

---

## Numbers to lock (say cleanly)

| Fact | Value |
|------|--------|
| Dataset size | **307,511** applications |
| Default rate | **~8.07%** |
| Split | Train **196,806** · Val **49,202** · Test **61,503** |
| Features (after FE) | **140** |
| Production model | **XGBoost** (tuned) |
| Challenger | Logistic Regression · Baseline Dummy |
| Threshold τ | **0.51** (middle_ground, validation only) |
| Test accuracy | **86.2%** |
| Test recall | **40.1%** |
| Test precision | **26.5%** |
| Balanced accuracy | **65.2%** |
| ROC-AUC | **0.769** |
| PR-AUC | **0.261** |
| Lift | **3.29×** |
| Dummy ROC-AUC | **~0.50** |
| XGB trees | **350** · depth **6** · lr **0.05** · scale_pos_weight **~5.69** |

**Decision rule:** if \(P(\text{default}) \ge 0.51\) → **DECLINE**, else **APPROVE**.

---

# THE SCRIPT

### [0:00–0:40] Opening — problem & stakes

> Good morning. Today I’ll present an end-to-end **Credit Risk Assessment System**.
>
> The problem is practical and unforgiving: given only an applicant’s profile, should a lender **approve** credit—or **decline**—and be able to say **why**?
>
> I built this on the **Home Credit Default Risk** dataset—**three hundred and seven thousand, five hundred and eleven** real applications. The target is whether the client later faced **payment difficulties**. Only about **eight percent** default. That class imbalance is exactly why reporting accuracy alone is misleading—and why this project had to be done carefully.
>
> The system scores **probability of default**, maps it to **approve or decline**, **explains** the decision with SHAP, and ships as a **Streamlit dashboard** and a **FastAPI** service.

*(Gesture to the dashboard hero.)*

---

### [0:40–1:35] Data, pipeline, models — sound rigorous

> On the technical side: this is a **Python** tabular ML stack—**pandas** and **numpy** for data, **scikit-learn** pipelines for preprocessing, **XGBoost** as the production model, **logistic regression** as a strong linear challenger, and a **stratified dummy** baseline so we know we’re beating chance.
>
> Preprocessing is inside a **Pipeline** fitted on train only—no leakage. Numerics get **median imputation**; categoricals get **most-frequent imputation** and **one-hot encoding**. Logistic regression also gets **standard scaling**; trees do not need it. On top of raw fields I engineered **domain ratios**—credit to income, annuity to income—plus **external score aggregates** and a clean handling of the known employment-day anomaly.
>
> The full academic split is stratified: about **one hundred ninety-seven thousand** train, **forty-nine thousand** validation, **sixty-one thousand** test. Models and the decision threshold are selected on **validation**. Final numbers I report are on **test only**.
>
> Imbalance is handled with **class_weight balanced** for logistic regression and **scale_pos_weight** near **five point seven** for XGBoost—not with a careless accuracy-only objective.
>
> One clarification students often ask: **there are no neural-net epochs**. XGBoost uses **three hundred fifty** boosted trees, max depth **six**, learning rate **zero point zero five**—that is our “training budget,” not deep learning.

---

### [1:35–2:20] Threshold policy — the advanced punch

> The model outputs **P of default**. A threshold **tau** turns probability into a business action: if **P is greater than or equal to tau**, we **decline**; otherwise we **approve**.
>
> Tau is a **policy**, not a second model. I compared policies honestly on validation. A **Youden** cut catches many defaulters but accuracy falls near **sixty-nine percent**. A **max-accuracy** cut can push accuracy near **ninety-two percent**, but recall collapses to about **seven percent**—almost never finding defaulters. That would flatter a slide and fail as risk management.
>
> Production uses a **middle-ground** policy: on validation, require accuracy at least **eighty percent** and recall at least **thirty-five percent**, then maximize **F1**. That freezes **tau at zero point five one**.
>
> On the held-out test set that yields **eighty-six point two percent accuracy**, **forty percent recall**, **twenty-seven percent precision**, **balanced accuracy about sixty-five percent**, and lift of **three point three**—meaning people we decline default over **three times** more often than the eight percent base rate. **ROC-AUC is zero point seven six nine** versus a dummy near **point five zero**. Changing tau does **not** retrain the model; ROC-AUC stays the same. That separation of **ranking skill** and **decision policy** is deliberate.

---

### [2:20–3:30] Live demo — three applicants

*(Stay on **Score applicant**. τ = 0.51.)*

> Let me show the same model on three different humans.

**1) Preset: Low risk profile → Run assessment**

> First, a strong profile—higher income, modest loan, high external scores, stable employment.  
> **P of default** is only about **two percent**—far below tau. Decision: **approve**.  
> Automation should say yes when risk is clearly low.

**2) Preset: Mid risk profile → Run assessment**

> Second, a middle profile—more leverage, mid-range external scores. Probability sits near **forty-eight percent**—still under tau, so **approve**, but you can *see* the boundary.  
> *(Optional, 10 sec: drag τ toward 0.40 and re-run.)* The **same applicant** can flip to decline. That is **risk appetite**, not a new model.  
> *(Reset τ to 0.51.)*

**3) Preset: High risk profile → Run assessment**

> Third, a stressed profile—young, short employment, large credit versus small income, external scores near **zero point one**.  
> Probability rises to about **sixty-seven percent**—above tau. Decision: **decline**—no loan.  
> We do not stop at a red banner. **SHAP** shows which features **raised** default risk for this person. That is the difference between a black box and an **explainable** decision aid—critical if anyone asks “why was I declined?”

---

### [3:30–4:20] Metrics tab — prove it is not cherry-picked

*(Open **Model metrics**.)*

> Scoring one person is not enough. Here is performance on **sixty-one thousand** held-out test applications.
>
> Production **XGBoost**: **accuracy eighty-six percent**, **recall forty percent**, **precision twenty-seven percent**, **ROC-AUC zero point seven seven**, lift about **three point three**.  
> The comparison table shows **dummy**, **logistic regression**, and **XGBoost** at the **same** threshold. Dummy ranks at chance. Logistic regression is a credible linear baseline—ROC-AUC about **zero point seven five**. XGBoost is production.
>
> The policy comparison table shows Youden, middle-ground, and max-accuracy side by side. We publish the extremes so we are not accused of hiding tradeoffs. Impressive systems report the number that **honest**—not only the number that flatters.

*(Optional 5 sec: switch sidebar **Score with** to Logistic Regression so headlines change, then back to Production.)*

> The dashboard is a lab: model choice and threshold are live controls.

---

### [4:20–4:50] Engineering close — sound senior

> Under the hood: **scikit-learn Pipeline** for train-serve consistency, **joblib** artifacts, **FastAPI** and **Uvicorn** for the API, **Streamlit** and **Plotly** for this UI, **pytest** for regression—including a **thousand-case spectrum suite** where every decision is checked against the threshold rule and designed risk tracks predicted risk very tightly.
>
> Stack in one breath: **Python, pandas, numpy, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, joblib, pytest**—on full Home Credit application data, with documented operating policy and no test-set shopping for tau.

---

### [4:50–5:05] Closing

> To summarize: full-data default risk modeling on **three hundred seven thousand** applications; rigorous train-validation-test discipline; imbalance-aware training; a transparent **middle-ground** threshold at **zero point five one**; test **accuracy eighty-six percent**, **recall forty percent**, **ROC-AUC zero point seven seven**; live approve-decline with **SHAP**; and production-style serving.
>
> Happy to take questions on threshold policy, metrics under imbalance, or multi-table extensions. Thank you.

*(Stop. Smile. Wait.)*

---

# Backup lines (if asked mid-talk)

**“Why not 95% accuracy?”**  
> Always-approve is about **ninety-two percent** accurate and catches **zero** defaulters. We keep **eighty-six percent** accuracy with **forty percent** recall.

**“Precision only 27%?”**  
> Base rate is **eight percent**. Lift **three point three** means the declined group is over three times riskier than average.

**“Epochs?”**  
> Not a neural net—**three hundred fifty** XGBoost trees.

**“Did you use the test set to pick tau?”**  
> No. Validation only; test is report-only.

**“Is this bank-ready?”**  
> Educational system—not a licensed underwriting product. Real banks need monitoring, governance, and fair-lending review.

**“Tech stack?”**  
> Python · pandas/numpy · scikit-learn · XGBoost · SHAP · FastAPI/Uvicorn · Streamlit/Plotly · joblib · pytest.

---

# Live click checklist

```
[ ] Dashboard open, Production model, τ = 0.51
[ ] 0:00  Problem + 307,511 + 8% imbalance
[ ] 0:40  Pipeline + models + 350 trees (no epochs)
[ ] 1:35  Middle-ground τ = 0.51 + metrics story
[ ] 2:20  Low preset  → APPROVE (~2%)
[ ] 2:50  Mid preset  → APPROVE near boundary (~48%)
[ ] 3:10  High preset → DECLINE (~67%) + SHAP
[ ] 3:30  Metrics tab — 86% / 40% / 0.769
[ ] 4:20  Stack + tests
[ ] 4:50  Close
```

---

# Optional extra demo profiles (if time)

| Profile | Expect | Approx P |
|---------|--------|----------|
| Low preset | Approve | ~2% |
| Mid preset | Approve | ~48% |
| Borderline (income 90k, credit 520k, EXT ~0.3) | Decline | ~59% |
| High preset | Decline | ~67% |
| Extreme (income 30k, credit 1.1M, EXT ~0.05) | Decline | ~79% |

---

# Delivery notes

1. Say **“eighty-six point two percent”**, not “about eighty-six.”  
2. Pause half a second on **APPROVE** / **DECLINE**.  
3. Call τ **policy**; call XGBoost **model**.  
4. Never apologize for precision—lead with **base rate + lift**.  
5. If you run long: cut mid τ-slider and LR sidebar switch.  
6. If you run short: add Extreme risk profile + one fairness disclaimer.

---

**Run dashboard**

```powershell
cd "~/credit-risk-assessment"
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Open: **http://localhost:8501**
