# Spectrum 1000 test report

- Cases: **1000**
- Model: **xgboost**
- Threshold τ: **0.5100**
- Logic failures: **0** → ALL PASSED
- Approve / Decline: **764** / **236** (decline rate 23.6%)
- Safest pole: p=0.0317 → **approve**
- Riskiest pole: p=0.7828 → **decline**
- Mean P low band (≤0.2): 0.0290
- Mean P high band (≥0.8): 0.7114
- Decline rate low (≤0.25): 0.0%
- Decline rate high (≥0.75): 91.2%
- Spearman(risk_level, P): **0.978** (p=0.00e+00)

Artifacts: `spectrum_1000_results.json`, `spectrum_1000_cases.json`