"""
WAZE CARGO — Step 04: Port Congestion ML Pipeline (Hybrid Ensemble)
====================================================================
Reads from:   RDS  maritime.clean_maritime_imports / exports
Writes to:    RDS  ml.port_monthly_agg          (monthly aggregation)
                   ml.port_features_indexed     (feature panel + COVID flags)
                   ml.port_cv_metrics           (walk-forward CV per model/port/fold)
                   ml.port_model_selection      (hybrid: best model per port)
                   ml.port_forecast_2026        (12-month forecast per port)

Hybrid ensemble strategy (from notebook 11):
  Big ports   (>= 500 ships/mo avg) -> Baseline Seasonal Naive
  Small ports (< 500 ships/mo avg)  -> Best ML model per port (by CV)

Models evaluated: Baseline, LightGBM, XGBoost, Random Forest, Ridge,
                  Lasso, ElasticNet

RUN:
    export RDS_HOST=... RDS_USER=... RDS_PASSWORD=... RDS_DBNAME=...
    python3 04_ml_congestion.py
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
log = logging.getLogger("ml_congestion")

# ── Config ────────────────────────────────────────────────────────
SRC = "maritime"
S   = "ml"
VOLUME_THRESHOLD = 500

LGBM_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.05,
    num_leaves=31, min_child_samples=10, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, verbose=-1, n_jobs=-1)
LGBM_ROUNDS = 500
LGBM_EARLY  = 50

XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=50, eval_metric="rmse",
    verbosity=0, n_jobs=-1)

RF_PARAMS = dict(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=42)


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
    engine = create_engine(
        f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{db}?sslmode=require",
        pool_pre_ping=True, connect_args={"connect_timeout": 15})
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
#  SQL — STEP 1: Monthly Aggregation
# ════════════════════════════════════════════════════════════════
SQL_MONTHLY_AGG = f"""
DROP TABLE IF EXISTS {S}.port_monthly_agg CASCADE;
CREATE TABLE {S}.port_monthly_agg AS

SELECT
    i.periodo::INTEGER AS year, i.mes::INTEGER AS month,
    i.puerto_desembarque AS port, 'import'::TEXT AS direction,
    COUNT(*)::BIGINT AS shipment_count,
    COALESCE(SUM(i.cif_us), 0) AS total_value_usd,
    NULL::DOUBLE PRECISION AS total_weight_mt,
    COALESCE(SUM(i.cantidad_mercancia), 0) AS total_quantity,
    COUNT(DISTINCT i.hs2_capitulo)::BIGINT AS commodity_diversity,
    COUNT(DISTINCT i.hs4_partida)::BIGINT AS hs4_diversity,
    COUNT(DISTINCT i.pais_origen)::BIGINT AS country_diversity,
    COUNT(DISTINCT i.continente_origen)::BIGINT AS continent_diversity,
    MODE() WITHIN GROUP (ORDER BY i.hs2_capitulo) AS dominant_hs2,
    MODE() WITHIN GROUP (ORDER BY i.tipo_carga) AS dominant_cargo_type,
    MODE() WITHIN GROUP (ORDER BY i.pais_origen) AS dominant_origin_country,
    MODE() WITHIN GROUP (ORDER BY i.continente_origen) AS dominant_continent,
    SUM(CASE WHEN i.tipo_carga ILIKE '%GENERAL%'    THEN 1 ELSE 0 END)::BIGINT AS cnt_general,
    SUM(CASE WHEN i.tipo_carga ILIKE '%GRANEL%'     THEN 1 ELSE 0 END)::BIGINT AS cnt_bulk,
    SUM(CASE WHEN i.tipo_carga ILIKE '%FRIGORI%'    THEN 1 ELSE 0 END)::BIGINT AS cnt_refrigerated,
    SUM(CASE WHEN i.tipo_carga ILIKE '%CONTENEDOR%' THEN 1 ELSE 0 END)::BIGINT AS cnt_container,
    AVG(i.cif_us) AS avg_value_per_shipment_usd
FROM {SRC}.clean_maritime_imports i
WHERE i.puerto_desembarque IS NOT NULL AND i.periodo BETWEEN 2005 AND 2025
GROUP BY i.puerto_desembarque, i.periodo, i.mes

UNION ALL

SELECT
    e.periodo::INTEGER, e.mes::INTEGER,
    e.puerto_embarque AS port, 'export'::TEXT,
    COUNT(*)::BIGINT,
    COALESCE(SUM(e.fob_us), 0),
    COALESCE(SUM(e.peso_bruto_kg) / 1000.0, 0) AS total_weight_mt,
    COALESCE(SUM(e.cantidad_mercancia), 0),
    COUNT(DISTINCT e.hs2_capitulo)::BIGINT,
    COUNT(DISTINCT e.hs4_partida)::BIGINT,
    COUNT(DISTINCT e.pais_destino)::BIGINT,
    COUNT(DISTINCT e.continente_destino)::BIGINT,
    MODE() WITHIN GROUP (ORDER BY e.hs2_capitulo),
    MODE() WITHIN GROUP (ORDER BY e.tipo_carga),
    MODE() WITHIN GROUP (ORDER BY e.pais_destino),
    MODE() WITHIN GROUP (ORDER BY e.continente_destino),
    SUM(CASE WHEN e.tipo_carga ILIKE '%GENERAL%'    THEN 1 ELSE 0 END)::BIGINT,
    SUM(CASE WHEN e.tipo_carga ILIKE '%GRANEL%'     THEN 1 ELSE 0 END)::BIGINT,
    SUM(CASE WHEN e.tipo_carga ILIKE '%FRIGORI%'    THEN 1 ELSE 0 END)::BIGINT,
    SUM(CASE WHEN e.tipo_carga ILIKE '%CONTENEDOR%' THEN 1 ELSE 0 END)::BIGINT,
    AVG(e.fob_us)
FROM {SRC}.clean_maritime_exports e
WHERE e.puerto_embarque IS NOT NULL AND e.periodo BETWEEN 2005 AND 2025
GROUP BY e.puerto_embarque, e.periodo, e.mes;

CREATE INDEX ON {S}.port_monthly_agg (port, direction, year, month);
"""

# ════════════════════════════════════════════════════════════════
#  SQL — STEP 2: Feature Engineering (with COVID-aware columns)
# ════════════════════════════════════════════════════════════════
SQL_FEATURES = f"""
DROP TABLE IF EXISTS {S}.port_features_indexed CASCADE;
CREATE TABLE {S}.port_features_indexed AS
WITH base AS (
    SELECT *,
        CASE WHEN month IN (12,1,2) THEN 'Summer'
             WHEN month IN (3,4,5)  THEN 'Autumn'
             WHEN month IN (6,7,8)  THEN 'Winter'
             ELSE 'Spring' END                                     AS season,
        ((month-1)/3+1)::DOUBLE PRECISION                          AS quarter,
        (year-2005)::INTEGER                                       AS year_index,
        SIN(2*PI()*month/12.0)                                     AS month_sin,
        COS(2*PI()*month/12.0)                                     AS month_cos,
        cnt_general*1.0/NULLIF(shipment_count,0)                   AS pct_general,
        cnt_bulk*1.0/NULLIF(shipment_count,0)                      AS pct_bulk,
        cnt_refrigerated*1.0/NULLIF(shipment_count,0)              AS pct_refrigerated,
        cnt_container*1.0/NULLIF(shipment_count,0)                 AS pct_container,
        COALESCE(total_weight_mt,0)/NULLIF(shipment_count,0)       AS weight_per_shipment_mt,
        total_quantity/NULLIF(shipment_count,0)                     AS avg_quantity_per_shipment,
        LAG(shipment_count, 1) OVER w   AS lag_1,
        LAG(shipment_count, 2) OVER w   AS lag_2,
        LAG(shipment_count, 3) OVER w   AS lag_3,
        LAG(shipment_count,12) OVER w   AS lag_12,
        LAG(shipment_count,24) OVER w   AS _lag_24,
        LAG(shipment_count,36) OVER w   AS _lag_36,
        LAG(shipment_count,48) OVER w   AS _lag_48,
        LAG(total_value_usd,12) OVER w  AS lag_value_12,
        LAG(total_weight_mt,12) OVER w  AS lag_weight_12,
        AVG(shipment_count::DOUBLE PRECISION)
            OVER (PARTITION BY port,direction ORDER BY year,month
                  ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)       AS rolling_3_mean,
        AVG(shipment_count::DOUBLE PRECISION)
            OVER (PARTITION BY port,direction ORDER BY year,month
                  ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)      AS rolling_12_mean,
        AVG(total_value_usd)
            OVER (PARTITION BY port,direction ORDER BY year,month
                  ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)      AS rolling_value_12_mean,
        (shipment_count - LAG(shipment_count,12) OVER w)::DOUBLE PRECISION
            / NULLIF(LAG(shipment_count,12) OVER w, 0)             AS yoy_growth,
        (total_value_usd - LAG(total_value_usd,12) OVER w)
            / NULLIF(LAG(total_value_usd,12) OVER w, 0)           AS yoy_value_growth,
        CASE WHEN year=2020 THEN 1 ELSE 0 END                     AS is_covid_shock,
        CASE WHEN year=2021 THEN 1 ELSE 0 END                     AS is_covid_rebound,
        CASE WHEN year=2022 THEN 1 ELSE 0 END                     AS is_covid_aftershock,
        CASE WHEN LAG(year,12) OVER w IN (2020,2021,2022)
             THEN 1 ELSE 0 END                                     AS lag_12_is_covid
    FROM {S}.port_monthly_agg
    WINDOW w AS (PARTITION BY port, direction ORDER BY year, month)
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
),
with_norm AS (
    SELECT *,
        (shipment_count - MIN(shipment_count) OVER pd)
            / NULLIF(MAX(shipment_count) OVER pd - MIN(shipment_count) OVER pd, 0) AS sc_norm,
        (total_value_usd - MIN(total_value_usd) OVER pd)
            / NULLIF(MAX(total_value_usd) OVER pd - MIN(total_value_usd) OVER pd, 0) AS v_norm,
        (COALESCE(total_weight_mt,0) - MIN(COALESCE(total_weight_mt,0)) OVER pd)
            / NULLIF(MAX(COALESCE(total_weight_mt,0)) OVER pd
                   - MIN(COALESCE(total_weight_mt,0)) OVER pd, 0) AS w_norm,
        (commodity_diversity - MIN(commodity_diversity) OVER pd)
            / NULLIF(MAX(commodity_diversity) OVER pd
                   - MIN(commodity_diversity) OVER pd, 0)          AS cd_norm
    FROM with_covid
    WINDOW pd AS (PARTITION BY port, direction)
)
SELECT *,
    CASE direction
        WHEN 'import' THEN
            COALESCE(sc_norm,0)*0.40 + COALESCE(v_norm,0)*0.30
            + COALESCE(cd_norm,0)*0.20 + COALESCE(pct_container,0)*0.10
        ELSE
            COALESCE(w_norm,0)*0.40 + COALESCE(sc_norm,0)*0.30
            + COALESCE(v_norm,0)*0.20 + COALESCE(pct_refrigerated,0)*0.10
    END AS congestion_index
FROM with_norm;

CREATE INDEX ON {S}.port_features_indexed (port, direction, year, month);
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
    m = RandomForestRegressor(**RF_PARAMS)
    m.fit(tr[features].fillna(0).astype(float),
          tr[U.TARGET].astype(float).values)
    return np.maximum(1.0, m.predict(
        df_test[features].fillna(0).astype(float)))


def _make_linear_fp(ModelCV, **kw):
    def fp(df_train, df_test, features):
        tr = df_train[~df_train["year"].isin(U.COVID_YEARS)]
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
    log.info("  WAZE CARGO  -  Hybrid Ensemble Congestion ML")
    log.info("=" * 60)

    # ── 0. Setup schema, drop old statistical tables ──────────
    with engine.begin() as c:
        c.execute(text(f"CREATE SCHEMA IF NOT EXISTS {S}"))
        for old in ["port_seasonal_index", "port_forecast_params",
                     "port_congestion_forecast"]:
            c.execute(text(f"DROP TABLE IF EXISTS {S}.{old} CASCADE"))
    log.info("Schema '%s' ready, old statistical tables dropped", S)

    # ── 1. Monthly aggregation ────────────────────────────────
    log.info("Step 1/5 - Building port_monthly_agg ...")
    run_sql(engine, "built -> port_monthly_agg", SQL_MONTHLY_AGG)

    # ── 2. Feature engineering ────────────────────────────────
    log.info("Step 2/5 - Building port_features_indexed ...")
    run_sql(engine, "built -> port_features_indexed", SQL_FEATURES)

    # ── 3. Walk-forward CV ────────────────────────────────────
    log.info("Step 3/5 - Walk-forward CV across all models ...")
    with engine.connect() as c:
        df_panel = pd.read_sql(
            text(f"SELECT * FROM {S}.port_features_indexed"), c)
    df_panel = (df_panel.sort_values(["port", "direction", "year", "month"])
                        .reset_index(drop=True))
    log.info("  Loaded %d rows, %d port-direction pairs",
             len(df_panel), df_panel.groupby(["port", "direction"]).ngroups)

    models = get_cv_models()
    log.info("  Models: %s", ", ".join(models.keys()))

    all_metrics = []
    for name, fp_fn in models.items():
        t0 = time.time()
        m = U.evaluate_model_across_ports(df_panel, fp_fn, name)
        log.info("    %-40s %4d scores  %.1fs",
                 name, len(m), time.time() - t0)
        all_metrics.append(m)
    cv_metrics = pd.concat(all_metrics, ignore_index=True)
    log.info("  Total CV scores: %d", len(cv_metrics))

    # ── 4. Hybrid model selection ─────────────────────────────
    log.info("Step 4/5 - Hybrid model selection ...")
    weights = (df_panel.groupby(["port", "direction"])[U.TARGET]
               .mean().rename("avg_volume").reset_index())

    cv_scores = (cv_metrics[cv_metrics["mape"].notna()]
                 .groupby(["port", "direction", "model"])["mape"]
                 .mean().reset_index()
                 .rename(columns={"mape": "cv_mape"}))
    cv_scores = cv_scores.merge(weights, on=["port", "direction"],
                                how="left")

    best_ml = cv_scores.loc[
        cv_scores.groupby(["port", "direction"])["cv_mape"].idxmin()
    ].copy()
    best_ml = best_ml.rename(columns={"model": "best_ml_model",
                                       "cv_mape": "ml_cv_mape"})

    best_ml["selected_model"] = np.where(
        best_ml["avg_volume"] >= VOLUME_THRESHOLD,
        "baseline_seasonal_naive_covid_aware",
        best_ml["best_ml_model"])

    baseline_cv = cv_scores[
        cv_scores["model"] == "baseline_seasonal_naive_covid_aware"
    ][["port", "direction", "cv_mape"]].rename(
        columns={"cv_mape": "baseline_cv_mape"})
    best_ml = best_ml.merge(baseline_cv, on=["port", "direction"],
                            how="left")
    best_ml["final_cv_mape"] = np.where(
        best_ml["avg_volume"] >= VOLUME_THRESHOLD,
        best_ml["baseline_cv_mape"], best_ml["ml_cv_mape"])

    n_big = (best_ml["avg_volume"] >= VOLUME_THRESHOLD).sum()
    n_small = (best_ml["avg_volume"] < VOLUME_THRESHOLD).sum()
    log.info("  Big ports (>=%d avg): %d -> Baseline",
             VOLUME_THRESHOLD, n_big)
    log.info("  Small ports:          %d -> CV-selected ML", n_small)
    for m, cnt in best_ml["selected_model"].value_counts().items():
        log.info("    %-35s %d ports", m, cnt)

    # ── 5. Generate 2026 forecasts ────────────────────────────
    log.info("Step 5/5 - Generating 2026 forecasts ...")
    eligible = U.list_eligible_ports(df_panel)
    sel_map = dict(zip(
        zip(best_ml["port"], best_ml["direction"]),
        best_ml["selected_model"]))

    t0 = time.time()
    all_fc = []
    for _, p in eligible.iterrows():
        port, direction = p["port"], p["direction"]
        model_name = sel_map.get((port, direction),
                                "baseline_seasonal_naive_covid_aware")
        df_port = U.get_port_panel(df_panel, port, direction)
        fit_fn, pred_fn = get_forecast_callbacks(model_name)
        fc = U.forecast_2026(df_port, fit_fn, pred_fn)
        if len(fc):
            fc = fc[fc["year"] == 2026].copy()
            fc["port"] = port
            fc["direction"] = direction
            fc["model"] = model_name
            all_fc.append(fc)

    forecast = (pd.concat(all_fc, ignore_index=True)
                if all_fc else pd.DataFrame())
    n_pairs = (forecast[["port", "direction"]].drop_duplicates().shape[0]
               if len(forecast) else 0)
    log.info("  Generated %d forecast rows for %d pairs in %.1fs",
             len(forecast), n_pairs, time.time() - t0)

    if len(forecast):
        annual = forecast.groupby("direction")["pred_shipment_count"].sum()
        log.info("  2026 totals - Imports: %s  Exports: %s",
                 f"{annual.get('import', 0):,.0f}",
                 f"{annual.get('export', 0):,.0f}")

    # ── Write results to ml schema ────────────────────────────
    log.info("Writing results to %s schema ...", S)
    now = datetime.now(timezone.utc)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.port_cv_metrics"))
        cv_out = cv_metrics.copy()
        cv_out["created_at"] = now
        cv_out.to_sql("port_cv_metrics", conn, schema=S,
                       index=False, method="multi", chunksize=500)
        n = conn.execute(text(
            f"SELECT COUNT(*) FROM {S}.port_cv_metrics")).scalar()
        log.info("  + %s.port_cv_metrics: %d rows", S, n)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.port_model_selection"))
        sel_out = best_ml[["port", "direction", "avg_volume",
                           "selected_model", "best_ml_model",
                           "ml_cv_mape", "baseline_cv_mape",
                           "final_cv_mape"]].copy()
        sel_out["volume_threshold"] = VOLUME_THRESHOLD
        sel_out["created_at"] = now
        sel_out.to_sql("port_model_selection", conn, schema=S,
                        index=False, method="multi")
        n = conn.execute(text(
            f"SELECT COUNT(*) FROM {S}.port_model_selection")).scalar()
        log.info("  + %s.port_model_selection: %d rows", S, n)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {S}.port_forecast_2026"))
        fc_out = forecast[["year", "month", "port", "direction",
                           "pred_shipment_count", "model"]].copy()
        fc_out["forecast_shipments"] = (
            fc_out["pred_shipment_count"].round(0).astype(int))
        fc_out["created_at"] = now
        fc_out.to_sql("port_forecast_2026", conn, schema=S,
                       index=False, method="multi")
        n = conn.execute(text(
            f"SELECT COUNT(*) FROM {S}.port_forecast_2026")).scalar()
        log.info("  + %s.port_forecast_2026: %d rows", S, n)

    engine.dispose()
    log.info("=" * 60)
    log.info("  Hybrid Ensemble ML complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
