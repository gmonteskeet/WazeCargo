All files live under **`/home/koko/Waze_Cargo/WazeCargo/notebooks/eda/`**:

## Notebooks (executed, with outputs embedded)

| File | Purpose |
|---|---|
| `01_eda_port_congestion.ipynb` | Exploratory data analysis (now includes §7.1 leak audit) |
| `02_baseline_seasonal_naive.ipynb` | COVID-aware seasonal-naive baseline |
| `03_lightgbm.ipynb` | LightGBM with SHAP, residuals, 2026 forecast |
| `04_xgboost.ipynb` | XGBoost equivalent |
| `05_random_forest.ipynb` | Random Forest equivalent |
| `06_linear_models.ipynb` | Ridge / Lasso / ElasticNet |
| `07_prophet.ipynb` | Prophet on the top 12 ports |
| `08_model_comparison_2026.ipynb` | Volume-weighted ranking + 2026 forecast |
| **`09_bugs_found.ipynb`** | **Failure log — both data-leakage bugs documented** |

## Shared Python utilities

- `wz_ml_utils.py` — shared `FEATURE_COLS`, walk-forward CV, scoring, `forecast_2026()`. Both leak fixes live here.
- `_nb_builder.py` — helper to build .ipynb files from cell tuples.
- `build_feature_cache.py` — pulls from DuckDB → parquet feature cache.
- `build_01_eda.py` … `build_09_bugs_found.py` — one builder per notebook (re-executable, so the notebooks are reproducible from source).

## Data artifacts (`data/`)

- `port_features_indexed.parquet` (2.4 MB, 21,013 rows × 55 cols) — the COVID-aware feature panel
- `port_monthly_agg.parquet` (754 KB) — raw monthly aggregates
- `forecast_2026.parquet` (5.8 KB) — 12-month 2026 forecast for top 15 port-direction pairs
- `meta.json` — constants used by builders
- `metrics/*.parquet` — one file per model with per-port-fold MAE/RMSE/MAPE/R²

## Figures (`figures/`)

42 PNG charts numbered by notebook section (01_*, 02_*, …, 80_model_ranking, 83_top8_2026_forecast, etc.) — embedded in the notebooks but also saved as standalone images you can drop directly into the thesis document.
