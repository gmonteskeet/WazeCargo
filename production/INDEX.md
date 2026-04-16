# Waze Cargo — Production Scripts & Notebooks Reference Index

**Generated**: 2026-04-16  
**Data source**: AWS RDS PostgreSQL (`wazecargo-db`, database `waze_cargo`)  
**Winner model**: Random Forest (volume-weighted R² = 0.765)

---

## SQL Scripts

| File | Description |
|------|-------------|
| `rebuild_clean_maritime.sh` | Bash script that rebuilds `waze_cargo.clean_maritime_imports` and `clean_maritime_exports` from `structured.all_imports/exports` + all `structured.lkp_*` lookup tables. Chunks inserts by year to avoid RDS disk pressure. Filters: maritime only (`COD_VIA_TRANSPORTE=1`), excludes airports (`tipo_puerto='Aeropuerto'` or name `ILIKE 'AEROP%'`). Handles European decimal format (`"336,00"`) via `SPLIT_PART`. |
| `rebuild_clean_maritime.sql` | Standalone SQL version of the same rebuild logic (single-transaction, for reference — use the `.sh` for production). |
| `03_ml_congestion.sql` | Pure SQL pipeline that builds 5 tables on RDS: `port_monthly_agg` → `port_features_indexed` → `port_seasonal_index` → `port_forecast_params` → `port_congestion_forecast`. Aggregates imports/exports by port×month, computes lag features, congestion index (weighted composite of normalized shipment count, value, weight, diversity), seasonal factors, and a 12-month parametric congestion forecast. **Bug-2 fixed**: rolling means now use `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` (past-only). |

## Python Scripts — Production ML

| File | Description |
|------|-------------|
| `03_ml_congestion.py` | Python (SQLAlchemy) wrapper that executes the same 5-step congestion pipeline as `03_ml_congestion.sql` on RDS. Requires env vars: `RDS_HOST`, `RDS_USER`, `RDS_PASSWORD`, `RDS_DBNAME`. Outputs: `port_monthly_agg`, `port_features_indexed`, `port_seasonal_index`, `port_forecast_params`, `port_congestion_forecast`. **Bug-2 fixed**. |
| `04_ml_commodity.py` | Commodity-port forecasting pipeline on RDS. Reads `clean_maritime_exports`, builds: `commodity_port_params` (per HS2×port trend stats, ≥48 months history), `commodity_seasonal_index` (monthly FOB/weight seasonal factors), `commodity_port_forecast` (12-month forecast with season labels: High/Normal/Low). |

## Python Scripts — Feature Cache & Notebook Builders

| File | Description |
|------|-------------|
| `build_feature_cache.py` | Reads from RDS (no longer DuckDB), builds `port_monthly_agg.parquet` and `port_features_indexed.parquet` in `data/`. Adds COVID-aware features (shock/rebound/aftershock flags, clean lag-12, clean YoY growth). These parquet files feed all EDA notebooks. **Bug-2 fixed** in rolling means SQL. |
| `wz_ml_utils.py` | Shared ML utilities imported by every notebook. Defines: `FEATURE_COLS` (canonical feature list — rolling means re-included after Bug-2 fix, `sc_norm`/`v_norm`/`w_norm`/`cd_norm` still excluded for leakage), `walk_forward_eval()` (3-fold walk-forward CV: 2018→2019, 2019→2023, 2023→2025), `score()` (MAE/RMSE/MAPE/R²), `forecast_2026()` (12-step recursive), `summarise()` (volume-weighted aggregation), `save_metrics()`/`load_all_metrics()` (parquet metric persistence). COVID year down-weighting: 2020=0.1, 2021=0.2, 2022=0.4, 2024=0.0. |
| `_nb_builder.py` | Helper that converts `(cell_type, source)` lists into `.ipynb` files using `nbformat`. |

## Notebook Builders (build_NN_*.py → NN_*.ipynb)

| Builder | Notebook | Description |
|---------|----------|-------------|
| `build_01_eda.py` | `01_eda_port_congestion.ipynb` | Exploratory Data Analysis. Port traffic overview, seasonal decomposition, congestion index distribution, COVID impact visualization, top ports by volume, cargo type breakdown. 40 cells. |
| `build_02_baseline.py` | `02_baseline_seasonal_naive.ipynb` | **Baseline model**: Seasonal naive with COVID awareness. Predicts using lag-12 (same month last year), adjusted for COVID years. Sets the floor that ML models must beat. 21 cells. |
| `build_03_lightgbm.py` | `03_lightgbm.ipynb` | **LightGBM** gradient boosting. Walk-forward CV across all eligible ports (≥36 months history). Hyperparameters tuned for count time series. Feature importance analysis. 2026 recursive forecast. 27 cells. |
| `build_04_xgboost.py` | `04_xgboost.ipynb` | **XGBoost** gradient boosting. Same protocol as LightGBM for fair comparison. Walk-forward CV, feature importance, 2026 forecast. 17 cells. |
| `build_05_random_forest.py` | `05_random_forest.ipynb` | **Random Forest**. Same protocol. **Winner model** on volume-weighted R² (0.765). Walk-forward CV, feature importance, 2026 forecast. 16 cells. |
| `build_06_linear.py` | `06_linear_models.ipynb` | **Ridge, Lasso, ElasticNet**. Tests whether linear models can compete. After leakage fix, these score R²≈0.5-0.6 (honest). Previously showed R²=1.000 due to Bug-1 and Bug-2. 17 cells. |
| `build_07_prophet.py` | `07_prophet.ipynb` | **Facebook Prophet**. Univariate time series model (sees only date + target). Does not use engineered features. Poor performer on this panel (R²≈-0.1). 19 cells. |
| `build_08_comparison.py` | `08_model_comparison_2026.ipynb` | **Model comparison**. Aggregates all model metrics from `data/metrics/`, computes volume-weighted MAE/R², ranks models, plots side-by-side bar charts and fold-level heatmaps. Declares winner. 27 cells. |
| `build_09_bugs_found.py` | `09_bugs_found.ipynb` | **Data leakage post-mortem**. Documents Bug-1 (full-history min-max normalization) and Bug-2 (inclusive rolling means). Shows the closed-form identity that gave linear models R²=1.0, the asymmetric impact on trees, and the train/serve skew. Before-and-after metrics comparison. 17 cells. |

---

## Model Comparison Summary (2026-04-16, new clean data)

### Volume-weighted metrics (production-relevant)

| Rank | Model | wR² | wMAE (shipments/month) |
|------|-------|-----|------------------------|
| 1 | **Random Forest** | **0.765** | 872 |
| 2 | XGBoost | 0.750 | 838 |
| 3 | LightGBM | 0.721 | 839 |
| 4 | Lasso | 0.694 | 2,041 |
| 5 | Ridge | 0.681 | 2,058 |
| 6 | ElasticNet | 0.681 | 2,109 |
| 7 | Prophet | -0.235 | 7,544 |
| 8 | Baseline | -0.905 | 5,778 |

### Per-port median metrics

| Rank | Model | median R² | median MAE |
|------|-------|-----------|------------|
| 1 | Lasso | 0.601 | 3.5 |
| 2 | ElasticNet | 0.590 | 3.7 |
| 3 | XGBoost | 0.569 | 3.3 |
| 4 | Ridge | 0.518 | 3.8 |
| 5 | LightGBM | 0.511 | 3.7 |
| 6 | Random Forest | 0.458 | 3.8 |
| 7 | Prophet | -0.114 | 461.0 |
| 8 | Baseline | -0.321 | 4.3 |

### Key observations

- **Random Forest wins on volume-weighted R²** because it performs best on high-traffic ports (which dominate the weighted average). XGBoost is a close second.
- **Lasso leads per-port median R²** because it performs more consistently across small ports.
- **Linear models are now honest** (R²≈0.5-0.7) after leakage fix. Previously showed R²=1.000.
- **Rolling means re-included** as features after Bug-2 fix (past-only windows). This improved all tree models by ~0.15 R² compared to the excluded-features version.
- **Prophet underperforms** because it sees only univariate target + date, missing the rich feature set (lags, diversity, cargo mix, COVID flags).

---

## Database Tables (waze_cargo schema on RDS)

| Table | Rows | Built by |
|-------|------|----------|
| `clean_maritime_imports` | 14,568,825 | `rebuild_clean_maritime.sh` |
| `clean_maritime_exports` | 4,452,738 | `rebuild_clean_maritime.sh` |
| `port_monthly_agg` | 21,966 | `03_ml_congestion.py` |
| `port_features_indexed` | 20,916 | `03_ml_congestion.py` |
| `port_seasonal_index` | 791 | `03_ml_congestion.py` |
| `port_forecast_params` | 60 | `03_ml_congestion.py` |
| `port_congestion_forecast` | 720 | `03_ml_congestion.py` |
| `commodity_port_params` | 434 | `04_ml_commodity.py` |
| `commodity_seasonal_index` | 5,051 | `04_ml_commodity.py` |
| `commodity_port_forecast` | 5,208 | `04_ml_commodity.py` |

---

## How to reproduce

```bash
# 1. Set env vars
export RDS_HOST=wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com
export RDS_PORT=5432
export RDS_USER=postgresmasterWZ
export RDS_DBNAME=waze_cargo
export RDS_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id 'arn:aws:secretsmanager:eu-north-1:263704545424:secret:rds!db-a7a252bf-a8e7-4147-80c6-159d1f33846b-dlG7HK' \
  --query SecretString --output text | jq -rj '.password')

# 2. Rebuild clean tables (if source data changed)
bash rebuild_clean_maritime.sh

# 3. Run production ML
python 03_ml_congestion.py
python 04_ml_commodity.py

# 4. Rebuild feature cache for notebooks
cd <notebooks_eda_dir>
python build_feature_cache.py

# 5. Build + execute notebooks
for f in build_0*.py; do python "$f"; done
for nb in 0*.ipynb; do
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 "$nb" --output "$nb"
done
```
