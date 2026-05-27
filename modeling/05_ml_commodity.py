"""
WAZE CARGO — Step 05: Commodity ML Pipeline
============================================
Reads from:   RDS  maritime.clean_maritime_imports / exports
Writes to:    RDS  ml.commodity_monthly_agg       (HS2 x port x direction monthly)
                   ml.commodity_features          (feature panel + COVID flags)
                   ml.commodity_cv_metrics        (walk-forward CV per model/combo/fold)
                   ml.commodity_model_selection   (best model per HS2 x port x direction)
                   ml.commodity_forecast_2026     (12-month forecast per combo)

Tests 7 models per HS2 x port x direction combination:
  Baseline Seasonal Naive, LightGBM, XGBoost, Random Forest,
  Ridge, Lasso, ElasticNet

Best model selected per combination by lowest walk-forward CV MAPE.

RUN:
    export RDS_HOST=... RDS_USER=... RDS_PASSWORD=... RDS_DBNAME=...
    python3 05_ml_commodity.py
"""

import logging, os, sys, time, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wz_ml_utils as U

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    lgb = None
    HAS_LGBM = False

try:
    import xgboost as xgb_lib
    HAS_XGB = True
except ImportError:
    xgb_lib = None
    HAS_XGB = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ml_commodity")

# ── Config ────────────────────────────────────────────────────────
SRC = "maritime"
S   = "ml"
MIN_MONTHS = 36

LGBM_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.05,
    num_leaves=31, min_child_samples=10, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, verbose=-1, n_jobs=-1)
LGBM_ROUNDS = 300
LGBM_EARLY  = 50

XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=50, eval_metric="rmse",
    verbosity=0, n_jobs=-1)

RF_PARAMS = dict(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=42)

COMMODITY_FEATURES = [
    "lag_1", "lag_2", "lag_3",
    "lag_12_clean",
    "lag_value_12", "lag_weight_12",
    "yoy_growth_clean", "yoy_value_growth",
    "rolling_3_mean", "rolling_12_mean", "rolling_value_12_mean",
    "month_sin", "month_cos", "quarter", "year_index",
    "avg_value_per_shipment_usd",
    "weight_per_shipment_mt",
    "avg_quantity_per_shipment",
    "is_covid_shock", "is_covid_rebound", "is_covid_aftershock",
    "lag_12_is_covid",
]


# ── DB ────────────────────────────────────────────────────────────
def get_engine():
    h, p, u, pw, db = (os.environ.get(k, "") for k in
                        ["RDS_HOST", "RDS_PORT", "RDS_USER",
                         "RDS_PASSWORD", "RDS_DBNAME"])
    p = p or "5432"
    missing = [k for k, v in zip(
        ["RDS_HOST", "RDS_USER", "RDS_PASSWORD", "RDS_DBNAME"],
        [h, u, pw, db]) if not v]
    if missing:
        log.error("Missing env vars: %s", " ".join(missing))
        sys.exit(1)
    from sqlalchemy.engine import URL
    url = URL.create("postgresql+psycopg2",
                     username=u, password=pw,
                     host=h, port=int(p), database=db,
                     query={"sslmode": "require"})
    engine = create_engine(url, pool_pre_ping=True,
                           connect_args={"connect_timeout": 15})
    with engine.connect() as c:
        c.execute(text("SELECT 1"))
    log.info("Connected -> %s/%s", h, db)
    return engine


def run_sql(engine, label, sql):
    t0 = time.time()
    with engine.begin() as c:
        c.execute(text(sql))
    tbl = label.split("->")[-1].strip()
    try:
        with engine.connect() as c:
            n = c.execute(text(f"SELECT COUNT(*) FROM {S}.{tbl}")).scalar()
        log.info("  + %-40s  %9d rows  %.1fs", label, n, time.time() - t0)
    except Exception:
        log.info("  + %-40s  %.1fs", label, time.time() - t0)


# ════════════════════════════════════════════════════════════════
#  SQL — STEP 1: Commodity Monthly Aggregation
# ════════════════════════════════════════════════════════════════
SQL_COMMODITY_AGG = f"""
DROP TABLE IF EXISTS {S}.commodity_monthly_agg CASCADE;
CREATE TABLE {S}.commodity_monthly_agg AS

SELECT
    i.hs2_capitulo                                          AS hs2,
    i.puerto_desembarque                                    AS port,
    'import'::TEXT                                          AS direction,
    i.periodo::INTEGER                                      AS year,
    i.mes::INTEGER                                          AS month,
    COUNT(*)::BIGINT                                        AS shipment_count,
    COALESCE(SUM(i.cif_us), 0)                              AS total_value_usd,
    NULL::DOUBLE PRECISION                                  AS total_weight_mt,
    COALESCE(SUM(i.cantidad_mercancia), 0)                  AS total_quantity,
    MODE() WITHIN GROUP (ORDER BY i.descripcion_producto)   AS dominant_description
FROM {SRC}.clean_maritime_imports i
WHERE i.hs2_capitulo IS NOT NULL
  AND i.puerto_desembarque IS NOT NULL
  AND i.periodo BETWEEN 2005 AND 2025
GROUP BY i.hs2_capitulo, i.puerto_desembarque, i.periodo, i.mes

UNION ALL

SELECT
    e.hs2_capitulo,
    e.puerto_embarque,
    'export'::TEXT,
    e.periodo::INTEGER,
    e.mes::INTEGER,
    COUNT(*)::BIGINT,
    COALESCE(SUM(e.fob_us), 0),
    COALESCE(SUM(e.peso_bruto_kg) / 1000.0, 0),
    COALESCE(SUM(e.cantidad_mercancia), 0),
    MODE() WITHIN GROUP (ORDER BY e.descripcion_producto)
FROM {SRC}.clean_maritime_exports e
WHERE e.hs2_capitulo IS NOT NULL
  AND e.puerto_embarque IS NOT NULL
  AND e.periodo BETWEEN 2005 AND 2025
GROUP BY e.hs2_capitulo, e.puerto_embarque, e.periodo, e.mes;

CREATE INDEX ON {S}.commodity_monthly_agg (hs2, port, direction, year, month);
"""


# ════════════════════════════════════════════════════════════════
#  SQL — STEP 2: Commodity Feature Engineering (COVID-aware)
# ════════════════════════════════════════════════════════════════
SQL_COMMODITY_FEATURES = f"""
DROP TABLE IF EXISTS {S}.commodity_features CASCADE;
CREATE TABLE {S}.commodity_features AS
WITH base AS (
    SELECT *,
        ((month-1)/3+1)::DOUBLE PRECISION                          AS quarter,
        (year-2005)::INTEGER                                       AS year_index,
        SIN(2*PI()*month/12.0)                                     AS month_sin,
        COS(2*PI()*month/12.0)                                     AS month_cos,
        total_value_usd / NULLIF(shipment_count, 0)                AS avg_value_per_shipment_usd,
        COALESCE(total_weight_mt, 0) / NULLIF(shipment_count, 0)   AS weight_per_shipment_mt,
        total_quantity / NULLIF(shipment_count, 0)                  AS avg_quantity_per_shipment,
        LAG(shipment_count, 1) OVER w   AS lag_1,
        LAG(shipment_count, 2) OVER w   AS lag_2,
        LAG(shipment_count, 3) OVER w   AS lag_3,
        LAG(shipment_count,12) OVER w   AS lag_12,
        LAG(shipment_count,24) OVER w   AS _lag_24,
        LAG(shipment_count,36) OVER w   AS _lag_36,
        LAG(shipment_count,48) OVER w   AS _lag_48,
        LAG(total_value_usd,12) OVER w                             AS lag_value_12,
        LAG(COALESCE(total_weight_mt,0),12) OVER w                 AS lag_weight_12,
        AVG(shipment_count::DOUBLE PRECISION)
            OVER (PARTITION BY hs2, port, direction ORDER BY year, month
                  ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)        AS rolling_3_mean,
        AVG(shipment_count::DOUBLE PRECISION)
            OVER (PARTITION BY hs2, port, direction ORDER BY year, month
                  ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)       AS rolling_12_mean,
        AVG(total_value_usd)
            OVER (PARTITION BY hs2, port, direction ORDER BY year, month
                  ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)       AS rolling_value_12_mean,
        (shipment_count - LAG(shipment_count,12) OVER w)::DOUBLE PRECISION
            / NULLIF(LAG(shipment_count,12) OVER w, 0)             AS yoy_growth,
        (total_value_usd - LAG(total_value_usd,12) OVER w)
            / NULLIF(LAG(total_value_usd,12) OVER w, 0)           AS yoy_value_growth,
        CASE WHEN year=2020 THEN 1 ELSE 0 END                     AS is_covid_shock,
        CASE WHEN year=2021 THEN 1 ELSE 0 END                     AS is_covid_rebound,
        CASE WHEN year=2022 THEN 1 ELSE 0 END                     AS is_covid_aftershock,
        CASE WHEN LAG(year,12) OVER w IN (2020,2021,2022)
             THEN 1 ELSE 0 END                                     AS lag_12_is_covid
    FROM {S}.commodity_monthly_agg
    WINDOW w AS (PARTITION BY hs2, port, direction ORDER BY year, month)
),
filtered AS (
    SELECT * FROM base WHERE lag_1 IS NOT NULL AND lag_12 IS NOT NULL
),
with_covid AS (
    SELECT *,
        CASE
            WHEN lag_12_is_covid = 0 THEN lag_12
            WHEN (year-2) NOT IN (2020,2021,2022) AND _lag_24 IS NOT NULL THEN _lag_24
            WHEN (year-3) NOT IN (2020,2021,2022) AND _lag_36 IS NOT NULL THEN _lag_36
            WHEN (year-4) NOT IN (2020,2021,2022) AND _lag_48 IS NOT NULL THEN _lag_48
            ELSE lag_12
        END AS lag_12_clean,
        CASE
            WHEN lag_12_is_covid = 0 THEN yoy_growth
            ELSE (shipment_count - CASE
                    WHEN (year-2) NOT IN (2020,2021,2022) AND _lag_24 IS NOT NULL THEN _lag_24
                    WHEN (year-3) NOT IN (2020,2021,2022) AND _lag_36 IS NOT NULL THEN _lag_36
                    WHEN (year-4) NOT IN (2020,2021,2022) AND _lag_48 IS NOT NULL THEN _lag_48
                    ELSE lag_12
                  END)::DOUBLE PRECISION
                / NULLIF(CASE
                    WHEN (year-2) NOT IN (2020,2021,2022) AND _lag_24 IS NOT NULL THEN _lag_24
                    WHEN (year-3) NOT IN (2020,2021,2022) AND _lag_36 IS NOT NULL THEN _lag_36
                    WHEN (year-4) NOT IN (2020,2021,2022) AND _lag_48 IS NOT NULL THEN _lag_48
                    ELSE lag_12
                  END, 0)
        END AS yoy_growth_clean
    FROM filtered
)
SELECT * FROM with_covid;

CREATE INDEX ON {S}.commodity_features (hs2, port, direction, year, month);
"""


# ════════════════════════════════════════════════════════════════
#  MODEL CALLBACKS — Walk-forward CV
# ════════════════════════════════════════════════════════════════

def _baseline_fp(df_train, df_test, features):
    preds = []
    for _, row in df_test.iterrows():
        lag12 = row.get("lag_12_clean", row.get("lag_12", 1))
        growth = row.get("yoy_growth_clean", 0)
        if pd.isna(lag12): lag12 = 1
        if pd.isna(growth): growth = 0
        growth = max(-0.20, min(0.20, float(growth)))
        preds.append(max(1.0, float(lag12) * (1 + growth)))
    return np.array(preds)


def _lgbm_fp(df_train, df_test, features):
    clean_yrs = sorted(y for y in df_train["year"].unique()
                       if y not in U.COVID_YEARS)
    if len(clean_yrs) >= 2:
        val_yr = clean_yrs[-1]
        tr = df_train[df_train["year"] < val_yr]
        vl = df_train[df_train["year"] == val_yr]
    else:
        tr, vl = df_train.iloc[:-12], df_train.iloc[-12:]
    if len(tr) < 12 or len(vl) < 3:
        tr, vl = df_train.iloc[:-6], df_train.iloc[-6:]
    Xtr = tr[features].fillna(0).astype(float)
    ytr = tr[U.TARGET].astype(float).values
    wtr = U.get_sample_weights(tr)
    Xv  = vl[features].fillna(0).astype(float)
    yv  = vl[U.TARGET].astype(float).values
    d  = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    dv = lgb.Dataset(Xv, label=yv, reference=d)
    model = lgb.train(LGBM_PARAMS, d, num_boost_round=LGBM_ROUNDS,
                      valid_sets=[dv],
                      callbacks=[lgb.early_stopping(LGBM_EARLY, verbose=False),
                                 lgb.log_evaluation(-1)])
    return np.maximum(1.0, model.predict(
        df_test[features].fillna(0).astype(float)))


def _xgb_fp(df_train, df_test, features):
    clean_yrs = sorted(y for y in df_train["year"].unique()
                       if y not in U.COVID_YEARS)
    if len(clean_yrs) >= 2:
        val_yr = clean_yrs[-1]
        tr = df_train[df_train["year"] < val_yr]
        vl = df_train[df_train["year"] == val_yr]
    else:
        tr, vl = df_train.iloc[:-12], df_train.iloc[-12:]
    if len(tr) < 12 or len(vl) < 3:
        tr, vl = df_train.iloc[:-6], df_train.iloc[-6:]
    Xtr = tr[features].fillna(0).astype(float)
    ytr = tr[U.TARGET].astype(float).values
    wtr = U.get_sample_weights(tr)
    Xv  = vl[features].fillna(0).astype(float)
    yv  = vl[U.TARGET].astype(float).values
    m = xgb_lib.XGBRegressor(**XGB_PARAMS)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xv, yv)], verbose=False)
    return np.maximum(1.0, m.predict(
        df_test[features].fillna(0).astype(float)))


def _rf_fp(df_train, df_test, features):
    tr = df_train[~df_train["year"].isin(U.COVID_YEARS)]
    if len(tr) < 12:
        tr = df_train
    m = RandomForestRegressor(**RF_PARAMS)
    m.fit(tr[features].fillna(0).astype(float),
          tr[U.TARGET].astype(float).values)
    return np.maximum(1.0, m.predict(
        df_test[features].fillna(0).astype(float)))


def _make_linear_fp(ModelCV, **kw):
    def fp(df_train, df_test, features):
        tr = df_train[~df_train["year"].isin(U.COVID_YEARS)]
        if len(tr) < 12:
            tr = df_train
        p = Pipeline([("scale", StandardScaler()),
                      ("model", ModelCV(**kw))])
        p.fit(tr[features].fillna(0).astype(float),
              tr[U.TARGET].astype(float).values)
        return np.maximum(1.0, p.predict(
            df_test[features].fillna(0).astype(float)))
    return fp


def get_cv_models():
    models = {
        "baseline_seasonal_naive_covid_aware": _baseline_fp,
        "ridge":      _make_linear_fp(RidgeCV, alphas=[0.01, 0.1, 1, 10, 100]),
        "lasso":      _make_linear_fp(LassoCV, alphas=[0.01, 0.1, 1, 10, 100],
                                      max_iter=5000),
        "elasticnet": _make_linear_fp(ElasticNetCV,
                                      l1_ratio=[0.1, 0.5, 0.7, 0.9],
                                      alphas=[0.01, 0.1, 1, 10],
                                      max_iter=5000),
        "random_forest": _rf_fp,
    }
    if HAS_LGBM:
        models["lightgbm"] = _lgbm_fp
    if HAS_XGB:
        models["xgboost"] = _xgb_fp
    return models


# ════════════════════════════════════════════════════════════════
#  MODEL CALLBACKS — Forecast 2026
# ════════════════════════════════════════════════════════════════

def get_forecast_callbacks(name):
    """Return (fit_fn, predict_fn) for U.forecast_2026."""

    if name == "baseline_seasonal_naive_covid_aware":
        def fit_fn(dt, f): return "baseline"
        def pred_fn(model, df_row, f):
            l = float(df_row.get("lag_12_clean",
                                 df_row.get("lag_12", 1)).iloc[0])
            g = float(df_row.get("yoy_growth_clean",
                                 pd.Series([0])).iloc[0])
            if pd.isna(l): l = 1
            if pd.isna(g): g = 0
            return max(1.0, l * (1 + max(-0.20, min(0.20, g))))
        return fit_fn, pred_fn

    if name == "lightgbm":
        def fit_fn(dt, features):
            tr = dt[dt["year"] < 2025].copy()
            vl = dt[dt["year"] == 2025].copy()
            if len(vl) == 0: vl = tr.tail(12).copy()
            if len(tr) < 12: tr = dt.iloc[:-6].copy(); vl = dt.iloc[-6:].copy()
            d  = lgb.Dataset(tr[features].fillna(0).astype(float),
                             label=tr[U.TARGET].astype(float).values,
                             weight=U.get_sample_weights(tr))
            dv = lgb.Dataset(vl[features].fillna(0).astype(float),
                             label=vl[U.TARGET].astype(float).values,
                             reference=d)
            return lgb.train(LGBM_PARAMS, d, num_boost_round=LGBM_ROUNDS,
                             valid_sets=[dv],
                             callbacks=[lgb.early_stopping(LGBM_EARLY,
                                                           verbose=False),
                                        lgb.log_evaluation(-1)])
        def pred_fn(model, df_row, f):
            return float(model.predict(
                df_row[f].fillna(0).astype(float))[0])
        return fit_fn, pred_fn

    if name == "xgboost":
        def fit_fn(dt, features):
            tr = dt[dt["year"] < 2025].copy()
            vl = dt[dt["year"] == 2025].copy()
            if len(vl) == 0: vl = tr.tail(12).copy()
            if len(tr) < 12: tr = dt.iloc[:-6].copy(); vl = dt.iloc[-6:].copy()
            m = xgb_lib.XGBRegressor(**XGB_PARAMS)
            m.fit(tr[features].fillna(0).astype(float),
                  tr[U.TARGET].astype(float).values,
                  sample_weight=U.get_sample_weights(tr),
                  eval_set=[(vl[features].fillna(0).astype(float),
                             vl[U.TARGET].astype(float).values)],
                  verbose=False)
            return m
        def pred_fn(model, df_row, f):
            return float(model.predict(
                df_row[f].fillna(0).astype(float))[0])
        return fit_fn, pred_fn

    if name == "random_forest":
        def fit_fn(dt, features):
            tr = dt[~dt["year"].isin(U.COVID_YEARS)]
            if len(tr) < 12: tr = dt
            m = RandomForestRegressor(**RF_PARAMS)
            m.fit(tr[features].fillna(0).astype(float),
                  tr[U.TARGET].astype(float).values)
            return m
        def pred_fn(model, df_row, f):
            return float(model.predict(
                df_row[f].fillna(0).astype(float))[0])
        return fit_fn, pred_fn

    # Linear models
    if name == "ridge":
        Cls, kw = RidgeCV, dict(alphas=[0.01, 0.1, 1, 10, 100])
    elif name == "lasso":
        Cls, kw = LassoCV, dict(alphas=[0.01, 0.1, 1, 10, 100],
                                max_iter=5000)
    elif name == "elasticnet":
        Cls, kw = ElasticNetCV, dict(l1_ratio=[0.1, 0.5, 0.7, 0.9],
                                     alphas=[0.01, 0.1, 1, 10],
                                     max_iter=5000)
    else:
        raise ValueError(f"Unknown model: {name}")

    def fit_fn(dt, features):
        tr = dt[~dt["year"].isin(U.COVID_YEARS)]
        if len(tr) < 12: tr = dt
        p = Pipeline([("scale", StandardScaler()),
                      ("model", Cls(**kw))])
        p.fit(tr[features].fillna(0).astype(float),
              tr[U.TARGET].astype(float).values)
        return p
    def pred_fn(model, df_row, f):
        return float(model.predict(
            df_row[f].fillna(0).astype(float).values)[0])
    return fit_fn, pred_fn


# ════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main():
    engine = get_engine()

    log.info("=" * 60)
    log.info("  WAZE CARGO  -  Commodity ML Pipeline")
    log.info("=" * 60)

    # ── 0. Setup schema ──────────────────────────────────────────
    with engine.begin() as c:
        c.execute(text(f"CREATE SCHEMA IF NOT EXISTS {S}"))
    log.info("Schema '%s' ready", S)

    # ── 1. Commodity monthly aggregation ─────────────────────────
    log.info("Step 1/5 - Building commodity_monthly_agg ...")
    run_sql(engine, "built -> commodity_monthly_agg", SQL_COMMODITY_AGG)

    # ── 2. Feature engineering ───────────────────────────────────
    log.info("Step 2/5 - Building commodity_features ...")
    run_sql(engine, "built -> commodity_features", SQL_COMMODITY_FEATURES)

    # ── 3. Walk-forward CV ───────────────────────────────────────
    log.info("Step 3/5 - Walk-forward CV across all models ...")
    with engine.connect() as c:
        df_panel = pd.read_sql(
            text(f"SELECT * FROM {S}.commodity_features"), c)
    df_panel = (df_panel.sort_values(["hs2", "port", "direction", "year", "month"])
                        .reset_index(drop=True))

    groups = {k: v.sort_values(["year", "month"]).reset_index(drop=True)
              for k, v in df_panel.groupby(["hs2", "port", "direction"])
              if len(v) >= MIN_MONTHS}
    eligible = list(groups.keys())

    log.info("  Loaded %d rows, %d eligible combos (>= %d months)",
             len(df_panel), len(eligible), MIN_MONTHS)

    if not eligible:
        log.warning("  No eligible combos — aborting")
        engine.dispose()
        return

    models = get_cv_models()
    log.info("  Models: %s", ", ".join(models.keys()))

    all_metrics = []
    for name, fp_fn in models.items():
        t0 = time.time()
        count = 0
        for key in eligible:
            df_series = groups[key]
            for r in U.walk_forward_eval(df_series, fp_fn, COMMODITY_FEATURES):
                r["hs2"] = key[0]
                r["port"] = key[1]
                r["direction"] = key[2]
                r["model"] = name
                all_metrics.append(r)
            count += 1
        log.info("    %-40s %4d combos  %.1fs",
                 name, count, time.time() - t0)

    cv_metrics = pd.DataFrame(all_metrics)
    log.info("  Total CV scores: %d", len(cv_metrics))

    # ── 4. Model selection ───────────────────────────────────────
    log.info("Step 4/5 - Model selection ...")
    cv_valid = cv_metrics[cv_metrics["mape"].notna()].copy()

    if cv_valid.empty:
        log.warning("  No valid CV scores — skipping model selection")
        best = pd.DataFrame(columns=["hs2", "port", "direction",
                                      "selected_model", "cv_mape", "avg_volume"])
    else:
        cv_scores = (cv_valid.groupby(["hs2", "port", "direction", "model"])["mape"]
                     .mean().reset_index()
                     .rename(columns={"mape": "cv_mape"}))

        avg_vol = (df_panel.groupby(["hs2", "port", "direction"])[U.TARGET]
                   .mean().rename("avg_volume").reset_index())
        cv_scores = cv_scores.merge(avg_vol, on=["hs2", "port", "direction"],
                                    how="left")

        best = cv_scores.loc[
            cv_scores.groupby(["hs2", "port", "direction"])["cv_mape"].idxmin()
        ].copy()
        best = best.rename(columns={"model": "selected_model"})

        for direction in ["import", "export"]:
            dir_best = best[best["direction"] == direction]
            log.info("  %s: %d combos", direction.upper(), len(dir_best))
            for m, cnt in dir_best["selected_model"].value_counts().items():
                log.info("    %-35s %d", m, cnt)

    # ── 5. Generate 2026 forecasts ───────────────────────────────
    log.info("Step 5/5 - Generating 2026 forecasts ...")

    active_keys = set(
        df_panel[df_panel["year"] >= 2024]
        .groupby(["hs2", "port", "direction"])
        .size().index.tolist()
    )
    forecast_combos = [k for k in eligible if k in active_keys]
    log.info("  Active combos to forecast: %d / %d eligible",
             len(forecast_combos), len(eligible))

    desc_df = (df_panel.sort_values("year", ascending=False)
               .drop_duplicates(["hs2", "port", "direction"]))
    desc_map = {}
    for _, row in desc_df.iterrows():
        desc_map[(row["hs2"], row["port"], row["direction"])] = \
            row.get("dominant_description", "")

    sel_map = {}
    for _, row in best.iterrows():
        sel_map[(row["hs2"], row["port"], row["direction"])] = row["selected_model"]

    t0 = time.time()
    all_fc = []
    n_done = 0
    for key in forecast_combos:
        hs2, port, direction = key
        model_name = sel_map.get(key, "baseline_seasonal_naive_covid_aware")
        df_series = groups[key]

        try:
            fit_fn, pred_fn = get_forecast_callbacks(model_name)
            fc = U.forecast_2026(df_series, fit_fn, pred_fn, COMMODITY_FEATURES)
        except Exception as e:
            log.warning("  Forecast failed %s/%s/%s: %s",
                        hs2, port, direction, str(e)[:80])
            continue

        if len(fc):
            fc = fc[fc["year"] == 2026].copy()
            fc["hs2"] = hs2
            fc["port"] = port
            fc["direction"] = direction
            fc["model"] = model_name
            fc["commodity_description"] = desc_map.get(key, "")
            all_fc.append(fc)

        n_done += 1
        if n_done % 50 == 0:
            log.info("    %d / %d forecasts done ...",
                     n_done, len(forecast_combos))

    forecast = (pd.concat(all_fc, ignore_index=True)
                if all_fc else pd.DataFrame())
    n_combos = (forecast[["hs2", "port", "direction"]]
                .drop_duplicates().shape[0] if len(forecast) else 0)
    log.info("  Generated %d forecast rows for %d combos in %.1fs",
             len(forecast), n_combos, time.time() - t0)

    if len(forecast):
        annual = forecast.groupby("direction")["pred_shipment_count"].sum()
        log.info("  2026 totals - Imports: %s  Exports: %s",
                 f"{annual.get('import', 0):,.0f}",
                 f"{annual.get('export', 0):,.0f}")

    # ── Write results to ml schema ───────────────────────────────
    log.info("Writing results to %s schema ...", S)
    now = datetime.now(timezone.utc)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.commodity_cv_metrics"))
        if not cv_metrics.empty:
            cv_out = cv_metrics.copy()
            cv_out["created_at"] = now
            cv_out.to_sql("commodity_cv_metrics", conn, schema=S,
                           index=False, method="multi", chunksize=500)
        n = conn.execute(text(
            f"SELECT COUNT(*) FROM {S}.commodity_cv_metrics")).scalar()
        log.info("  + %s.commodity_cv_metrics: %d rows", S, n)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.commodity_model_selection"))
        if not best.empty:
            sel_out = best[["hs2", "port", "direction", "avg_volume",
                            "selected_model", "cv_mape"]].copy()
            sel_out["created_at"] = now
            sel_out.to_sql("commodity_model_selection", conn, schema=S,
                            index=False, method="multi")
        n = conn.execute(text(
            f"SELECT COUNT(*) FROM {S}.commodity_model_selection")).scalar()
        log.info("  + %s.commodity_model_selection: %d rows", S, n)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.commodity_forecast_2026"))
        if len(forecast):
            fc_out = forecast[["year", "month", "hs2", "port", "direction",
                               "pred_shipment_count", "model",
                               "commodity_description"]].copy()
            fc_out["forecast_shipments"] = (
                fc_out["pred_shipment_count"].round(0).astype(int))
            fc_out["created_at"] = now
            fc_out.to_sql("commodity_forecast_2026", conn, schema=S,
                           index=False, method="multi", chunksize=500)
        n = conn.execute(text(
            f"SELECT COUNT(*) FROM {S}.commodity_forecast_2026")).scalar()
        log.info("  + %s.commodity_forecast_2026: %d rows", S, n)

    engine.dispose()
    log.info("=" * 60)
    log.info("  Commodity ML Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
