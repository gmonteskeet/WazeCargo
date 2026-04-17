# Waze Cargo -- Production Scripts & Notebooks Reference Index

**Updated**: 2026-04-17
**Data source**: AWS RDS PostgreSQL (`wazecargo-db`, database `waze_cargo`)
**Winner model**: Baseline Seasonal Naive (COVID-aware) -- volume-weighted MAPE = 4.33%

---

## Directory Structure

```
production/
  03_ml_congestion.py          # RDS congestion pipeline
  03_ml_congestion.sql         # SQL reference version
  04_ml_commodity.py           # RDS commodity pipeline
  build_feature_cache.py       # RDS -> parquet cache
  build_01_eda.py              # Notebook builder: EDA
  build_02_baseline.py         # Notebook builder: Baseline
  build_03_lightgbm.py         # Notebook builder: LightGBM
  build_04_xgboost.py          # Notebook builder: XGBoost
  build_05_random_forest.py    # Notebook builder: Random Forest
  build_06_linear.py           # Notebook builder: Ridge/Lasso/ElasticNet
  build_07_prophet.py          # Notebook builder: Prophet
  build_08_comparison.py       # Notebook builder: Model Comparison + Chile Overview
  build_09_bugs_found.py       # Notebook builder: Data Leakage Post-mortem
  wz_ml_utils.py               # Shared ML utilities (features, CV, scoring, forecast)
  _nb_builder.py               # ipynb file generator
  notebooks/                   # Executed notebooks (canonical location)
    01_eda_port_congestion.ipynb
    02_baseline_seasonal_naive.ipynb
    03_lightgbm.ipynb
    04_xgboost.ipynb
    05_random_forest.ipynb
    06_linear_models.ipynb
    07_prophet.ipynb
    08_model_comparison_2026.ipynb
    09_bugs_found.ipynb
    figures/                   # All generated charts (PNG)
```

---

## SQL Scripts

| File | Description |
|------|-------------|
| `rebuild_clean_maritime.sh` | Rebuilds `waze_cargo.clean_maritime_imports/exports` from `structured.all_imports/exports` + lookup tables. Chunks by year to keep WAL bounded on db.t3.micro. |
| `rebuild_clean_maritime.sql` | Standalone SQL version (single-transaction, for reference). |
| `03_ml_congestion.sql` | Pure SQL pipeline: `port_monthly_agg` -> `port_features_indexed` -> `port_seasonal_index` -> `port_forecast_params` -> `port_congestion_forecast`. Bug-2 fixed (past-only rolling means). |

## Python Scripts -- Production ML

| File | Description |
|------|-------------|
| `03_ml_congestion.py` | SQLAlchemy wrapper for the 5-step congestion pipeline on RDS. Requires env vars: `RDS_HOST`, `RDS_USER`, `RDS_PASSWORD`, `RDS_DBNAME`. |
| `04_ml_commodity.py` | Commodity-port forecasting on RDS. Builds `commodity_port_params`, `commodity_seasonal_index`, `commodity_port_forecast`. |

## Python Scripts -- Feature Cache & Utilities

| File | Description |
|------|-------------|
| `build_feature_cache.py` | Reads from RDS, builds `port_monthly_agg.parquet` and `port_features_indexed.parquet` in `data/`. Adds COVID-aware features. |
| `wz_ml_utils.py` | Shared ML utilities: `FEATURE_COLS` (rolling means re-included after Bug-2 fix), `walk_forward_eval()` (3-fold CV), `score()`, `forecast_2026()` (12-step recursive), `summarise()` (volume-weighted). COVID weights: 2020=0.1, 2021=0.2, 2022=0.4, 2024=0.0. |
| `_nb_builder.py` | Converts `(cell_type, source)` lists into `.ipynb` files. |

## Notebooks (in `notebooks/`)

Each model notebook includes walk-forward CV metrics + a **top-8 ports by volume 2026 forecast chart**.

| Builder | Notebook | Description |
|---------|----------|-------------|
| `build_01_eda.py` | `01_eda_port_congestion.ipynb` | EDA: port traffic, seasonal decomposition, congestion index, COVID impact, cargo type breakdown. |
| `build_02_baseline.py` | `02_baseline_seasonal_naive.ipynb` | **Baseline**: seasonal naive + COVID awareness. Top-8 forecast chart. |
| `build_03_lightgbm.py` | `03_lightgbm.ipynb` | **LightGBM** gradient boosting. SHAP analysis + top-8 forecast chart. |
| `build_04_xgboost.py` | `04_xgboost.ipynb` | **XGBoost** gradient boosting. Feature importance + top-8 forecast chart. |
| `build_05_random_forest.py` | `05_random_forest.ipynb` | **Random Forest** (bagging). MDI importance + top-8 forecast chart. |
| `build_06_linear.py` | `06_linear_models.ipynb` | **Ridge, Lasso, ElasticNet**. Coefficient analysis + top-8 forecast chart (Ridge). |
| `build_07_prophet.py` | `07_prophet.ipynb` | **Prophet** (univariate). Component decomposition + top-8 forecast chart. |
| `build_08_comparison.py` | `08_model_comparison_2026.ipynb` | **Model comparison**: ranking, stability, head-to-head. **Chile-wide imports/exports overview** with all 6 models overlaid + zoom panels. |
| `build_09_bugs_found.py` | `09_bugs_found.ipynb` | Data leakage post-mortem (Bug-1: min-max, Bug-2: rolling means). |

---

## Model Comparison Summary (2026-04-17)

**Winner**: Baseline Seasonal Naive (COVID-aware) -- wMAPE 4.33%, wR2 0.392

### 2026 Annual Forecast Summary -- Total Chile Shipments

| Model | Imports | Exports | Total |
|-------|---------|---------|-------|
| **Baseline** | **856,350** | **186,148** | **1,042,499** |
| LightGBM | 744,916 | 183,383 | 928,299 |
| XGBoost | 740,204 | 177,017 | 917,221 |
| Random Forest | 805,354 | 178,314 | 983,668 |
| Ridge | 828,789 | 184,849 | 1,013,639 |
| Prophet | 783,781 | 175,596 | 959,377 |
| *Actual 2025* | *814,110* | *181,800* | *995,910* |

### Key observations

- **Baseline wins on wMAPE** because Chilean port traffic is highly seasonal and stable -- lag-12 with clean YoY growth captures the dominant signal.
- **Tree models (LightGBM, XGBoost, RF)** slightly underperform on wMAPE but add value on volatile small ports.
- **Feature engineering** (COVID-aware lags, sample weights) is more important than model choice.
- **Rolling means re-included** as features after Bug-2 fix (past-only windows).

---

## Database Tables (waze_cargo schema on RDS)

| Table | Built by |
|-------|----------|
| `clean_maritime_imports` | `rebuild_clean_maritime.sh` |
| `clean_maritime_exports` | `rebuild_clean_maritime.sh` |
| `port_monthly_agg` | `03_ml_congestion.py` |
| `port_features_indexed` | `03_ml_congestion.py` |
| `port_seasonal_index` | `03_ml_congestion.py` |
| `port_forecast_params` | `03_ml_congestion.py` |
| `port_congestion_forecast` | `03_ml_congestion.py` |
| `commodity_port_params` | `04_ml_commodity.py` |
| `commodity_seasonal_index` | `04_ml_commodity.py` |
| `commodity_port_forecast` | `04_ml_commodity.py` |

---

## How to reproduce

```bash
# 1. Set env vars
export RDS_HOST=wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com
export RDS_PORT=5432
export RDS_USER=postgresmasterWZ
export RDS_DBNAME=waze_cargo
export RDS_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id 'arn:aws:secretsmanager:eu-north-1:...' \
  --query SecretString --output text | jq -rj '.password')

# 2. Rebuild clean tables (if source data changed)
bash rebuild_clean_maritime.sh

# 3. Run production ML
python 03_ml_congestion.py
python 04_ml_commodity.py

# 4. Rebuild feature cache
python build_feature_cache.py

# 5. Build notebook skeletons
for f in build_0*.py; do python "$f"; done
mv 0*.ipynb notebooks/

# 6. Execute notebooks
cd notebooks
for nb in 0*.ipynb; do
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 "$nb" --output "$nb"
done
```
