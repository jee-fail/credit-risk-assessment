# Model artifacts

## Primary (default for API & dashboard)

**`full_data/`** — trained on entire `application_train` (307,511 rows)

| File | Role |
|------|------|
| `full_data/credit_risk_bundle.joblib` | Pipelines + metadata |
| `full_data/metrics.json` | Full evaluation (lifecycle, splits, lift, …) |
| `full_data/metrics_summary.md` | Human table |
| `full_data/README.md` | How to switch packs |

Production: **XGBoost** (hyperparameter-tuned), test ROC-AUC **0.769**, accuracy **86.2%**, recall **40.1%**, τ = **0.51** (`middle_ground` on validation). See `docs/OPERATING_POLICY.md`.

`config.py` selects this pack automatically when the joblib file exists.

## Comparison (not default)

| Path | Description |
|------|-------------|
| `credit_risk_bundle.joblib` + `metrics.json` | 50k stratified sample model |
| `*multitable_30k*` | 30k + multi-table feature research run |

## Env override

```powershell
# Full data (default if present)
$env:CREDIT_RISK_MODELS_DIR = "$PWD\models\full_data"

# Sample comparison pack
$env:CREDIT_RISK_MODELS_DIR = "$PWD\models"
```
