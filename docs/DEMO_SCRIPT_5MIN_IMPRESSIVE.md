# 5-Minute Impressive Demo Script  
## Credit Risk Assessment System

**Speaker:** Sudhanshu Kumar  
**Duration:** ~5:00 (tight — practice once with a timer)  
**Open before you start:** Dashboard http://localhost:8501 · optional `STATUS.md`

**Numbers to lock in memory**

| Item | Value |
|------|--------|
| Data | **307,511** applications |
| Default rate | **~8%** |
| Model | **XGBoost** (tuned) |
| Threshold τ | **0.51** (middle-ground policy) |
| Test accuracy | **86.2%** |
| Test recall | **40.1%** |
| Test precision | **26.5%** |
| ROC-AUC | **0.769** |
| Lift | **~3.3×** |
| Dummy AUC | **~0.50** |

---

## Stage setup (30 seconds before speaking)

1. Dashboard on **Score applicant**, sidebar **Score with = Production (xgboost)**, τ ≈ **0.51**.  
2. Do **not** run anything yet — click live during the talk.  
3. Have **Metrics** tab ready as second click.  
4. Stand slightly to the side of the screen; point with cursor, not finger on glass.

**Energy:** calm, confident, slightly fast on numbers, slower on the decision rule.

---

## FULL SCRIPT (~5:00)

### [0:00–0:45] Opening — problem & stakes

> “Good morning. I’m **Sudhanshu Kumar**, and this is my **Credit Risk Assessment System**.
>
> Every day, lenders face a hard binary decision: given only an application profile, should we **approve** credit—or **decline**—and **why**?
>
> I built an end-to-end machine learning system on the **Home Credit** default dataset—**three hundred and seven thousand** real applications. The label is simple and brutal: did the client later face **payment difficulties**? Only about **eight percent** do. That imbalance is what makes naive accuracy games useless—and what makes this problem interesting.
>
> My system does four things: it **scores** default risk, it turns that score into an **approve-or-decline decision**, it **explains** declines with SHAP, and it packages everything as a **production-style API and this interactive dashboard**.”

*(Gesture to the hero banner.)*

---

### [0:45–1:35] Academic ML lifecycle — sound rigorous

> “Before any demo click, the science. I followed a full academic lifecycle—not a notebook that overfits the test set.
>
> **Data and EDA** on the full train file. **Feature engineering**—income and credit ratios, external score aggregates, cleaning the famous `DAYS_EMPLOYED` anomaly. **Stratified train, validation, and test**—roughly one hundred ninety-seven thousand train, forty-nine thousand validation, sixty-one thousand test.
>
> On train I fit a **dummy baseline**, **logistic regression**, and **XGBoost**, with class imbalance handled via class weights and `scale_pos_weight`. **Hyperparameters** were searched on train and selected on **validation only**. The production model is **XGBoost**, test **ROC-AUC zero point seven six nine**—versus a dummy that sits at **point five zero**. That gap is real ranking skill.
>
> Then—and this matters—the **decision threshold** was also chosen on validation, never on test.”

---

### [1:35–2:15] The decision rule + operating policy (advanced, impressive)

> “The model outputs a probability of default—call it **P of default**. A threshold **tau** turns probability into a business action:
>
> If **P is greater than or equal to tau**, we **decline**. Otherwise we **approve**.
>
> Tau is not magic and it is not 0.5 by habit. I compared policies honestly. A **Youden** cut maximizes separation and catches many defaulters—but accuracy falls near **sixty-nine percent**. A **max-accuracy** cut pushes accuracy near **ninety-two percent**—but recall collapses to about **seven percent**, almost never catching defaulters. That would look good on a slide and fail as a risk system.
>
> So production uses a **middle-ground policy**: on validation, require accuracy at least **eighty percent** and recall at least **thirty-five percent**, then maximize **F1**. That freezes tau at **zero point five one**. On held-out test we get **eighty-six point two percent accuracy**, **forty percent recall**, **twenty-seven percent precision**, and about **three point three times lift** versus the eight percent base rate. ROC-AUC stays **zero point seven seven**—because changing tau does **not** retrain the model; it only changes the operating point.”

*(Optional one-liner, smile lightly:)*  
> “We publish the extreme policies in our docs so nobody can claim we hid the tradeoff.”

---

### [2:15–3:25] Live demo — spectrum of decisions

*(Dashboard: Score applicant.)*

> “Now the system in action—same model, three very different humans.”

**Click 1 — Low risk preset → Run assessment**

> “First, a **strong** applicant: higher income, modest credit, strong external scores, stable employment.  
> **P of default** is only about **two percent**—far below tau. Decision: **approve**.  
> This is what responsible automation should do: clear low risk, clear yes.”

**Click 2 — Mid risk preset → Run assessment**

> “Second, a **middle** profile—more leverage, mid external scores. Probability sits near **forty-eight percent**—still under tau, so **approve**, but you can *feel* the policy boundary.  
> If I drag tau down”—*(move threshold slider to ~0.40, re-run if needed)*—“the **same person** can flip to decline. That is not a new model. That is **risk appetite**. This is how you talk to a credit committee.”

*(Reset tau to **0.51**.)*

**Click 3 — High risk preset → Run assessment**

> “Third, a **stressed** profile: young, short employment, large loan versus small income, external scores near **zero point one**.  
> Probability jumps to about **sixty-seven percent**—above tau. Decision: **decline**—no loan.  
> And critically, we don’t stop at a red banner. We open **SHAP**: which features **raised** default risk for *this* person. That is the difference between a black-box score and an **explainable** underwriting assistant.”

*(Point at top SHAP bars—EXT_SOURCE / leverage themes if visible.)*

> “In regulated industries, ‘why’ is not optional. SHAP gives local, model-faithful reasons.”

---

### [3:25–4:20] Metrics tab + model comparison (advanced)

*(Open **Model metrics**.)*

> “Scoring is only half the story. Here is **held-out test performance** on sixty-one thousand applications the model never used for fitting or threshold choice.
>
> For production **XGBoost**: **accuracy eighty-six percent**, **recall forty percent**, **precision twenty-seven percent**, **ROC-AUC zero point seven seven**, lift about **three point three**.  
> Precision looks modest until you remember the base rate is **eight percent**—so we are concentrating risk at **three times** random screening.
>
> The table compares **dummy**, **logistic regression**, and **XGBoost** at the **same** threshold. Dummy ranks at chance. Logistic regression is a strong linear baseline. XGBoost wins on ranking and on our production accuracy.
>
> Sidebar **Score with** also switches metrics headlines—so this dashboard is not a static screenshot; it is a **lab**.”

*(Briefly switch Score with to Logistic Regression so headlines change, then back to Production.)*

> “Policy comparison shows Youden, middle-ground, and max-accuracy side by side. That transparency is deliberate. Impressive systems don’t only report the number that flatters them.”

---

### [4:20–4:50] Engineering & integrity (sound senior)

> “Under the hood: a **scikit-learn Pipeline**—imputation and encoding fitted on train only, no leakage. Artifacts shipped as a joblib bundle. **FastAPI** for programmatic assess and explain. **Streamlit** for interactive review. **Pytest** including a **thousand-case spectrum suite** where every decision is checked against the threshold rule, and designed risk levels correlate with predicted risk at Spearman nearly **zero point nine eight**.
>
> Reproducible seeds, documented operating policy, model card, and a full mentor Q-and-A bank. This is not a one-off accuracy chase—it is a **system**.”

---

### [4:50–5:00] Close

> “To summarize: full-data credit default modeling, rigorous train-val-test discipline, imbalance-aware training, an **ethical middle-ground threshold**, competitive **ROC-AUC zero point seven seven**, live approve-decline with **SHAP**, and production-style serving.
>
> I’m happy to go deeper on threshold policy, multi-table features, or fairness—thank you.”

*(Stop. Smile. Hands still. Wait for questions.)*

---

## If you run long (cuts, in order)

1. Skip logistic switch on Metrics.  
2. Skip tau slider mid demo.  
3. Shorten lifecycle to three sentences.  
4. Keep: problem → rule → low/high demo → metrics headline → close.

## If you run short (add-ons)

1. Show **Extreme risk** profile (P ~ **79%**, hard decline).  
2. Mention multi-table research pack as **next lift** on AUC.  
3. One fairness line: “Educational system—not production fair-lending certified.”

---

## Likely questions — 15-second answers

**“Why isn’t accuracy 95%?”**  
> “Always-approve is ~92% accurate and catches zero defaulters. We optimize a constrained tradeoff: 86% accuracy with 40% recall.”

**“Precision only 27%?”**  
> “Base rate 8%. Lift 3.3× means declined applicants default over three times more often than average.”

**“Did you leak the test set?”**  
> “No. Fit and threshold on train/val; test is report-only.”

**“Is this bank-ready?”**  
> “No—educational. Real deployment needs monitoring, governance, and fair-lending review.”

**“Why XGBoost?”**  
> “Best validation ROC-AUC versus logistic regression and dummy; strong tabular baseline with SHAP TreeExplainer.”

---

## Cursor checklist (print)

```
[ ] Dashboard up, production model, τ=0.51
[ ] 0:00  Problem + 307k + 8% imbalance
[ ] 0:45  Lifecycle + XGB AUC 0.769
[ ] 1:35  Tau policy middle-ground
[ ] 2:15  Low → approve
[ ] 2:40  Mid → near boundary (optional tau)
[ ] 3:00  High → decline + SHAP
[ ] 3:25  Metrics headlines
[ ] 4:20  Pipeline / API / 1000 tests
[ ] 4:50  Close
```

---

## Delivery tips (sound impressive)

1. **Say numbers cleanly:** “eighty-six point two percent,” not “about eighty-six.”  
2. **Pause half a second** after APPROVE / DECLINE banners.  
3. **Never apologize** for precision—lead with lift and base rate.  
4. **Use the word “policy”** for tau; “model” for XGBoost. Separating them sounds expert.  
5. **Eye contact on the close**, not the screen.
