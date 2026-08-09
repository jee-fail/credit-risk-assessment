# Product rules — credit card risk decisioning

## Business framing

The system supports a simplified **credit card underwriting** decision:

1. Estimate **probability of default** \(p = \hat{P}(\text{TARGET}=1 \mid x)\).  
2. If \(p \ge \tau\) → **decline**.  
3. If \(p < \tau\) → **approve**.

Home Credit’s `TARGET` measures **payment difficulties**, which we treat as the risk signal for the decision.

## Threshold \(\tau\)

- Default production policy: **`middle_ground`** on validation (accuracy ≥ 80% & recall ≥ 35%, max F1). Alternatives: Youden / F1 / fixed / max_accuracy via CLI or `scripts/retarget_threshold.py`. See `docs/OPERATING_POLICY.md`.  
- Request-time override: API field `threshold` (for demos and sensitivity analysis).  
- Stored permanently in `models/credit_risk_bundle.joblib` after training.

## Decline explanations

For declined applicants, the API returns SHAP-based top factors that **increased** predicted default risk, optionally mapped through `HomeCredit_columns_description.csv`.

## Explicit non-goals (v1)

- Live credit-bureau API pulls  
- Legally binding ECOA/FCRA adverse-action letters  
- LGD / EAD / limit assignment  
- User authentication and rate limiting  
