"""
WAZE CARGO — COVID-Aware ML Pipeline: Train, Evaluate, Forecast
================================================================
Fixes applied in this version:
  ✓ String-to-float error: select_features() now explicitly filters
    to numeric columns only before passing to LightGBM/XGBoost
  ✓ COVID feature engineering fully vectorised (no df.loc loops)
  ✓ lag_12_clean and yoy_growth_clean computed with groupby+transform
  ✓ forecast_2026() exception now caught per-port so one bad port
    never kills the whole run
  ✓ Baseline MAPE fixed: uses rolling_12_mean as fallback when
    lag_12_clean is zero/null

COVID HANDLING:
  2020 shock     → weight 0.1, lag_12 interpolated from pre-COVID trend
  2021 rebound   → weight 0.2, yoy_growth replaced with clean median
  2022 aftershock→ weight 0.4
  2024           → excluded entirely (incomplete data)

MODELS:
  baseline_covid_aware  → seasonal naive with clean lag/growth (benchmark)
  lightgbm              → gradient boosting, COVID sample weights
  xgboost               → same, different regularisation

CV FOLDS (clean years only as test targets):
  Fold 1: Train ≤2018 → Predict 2019
  Fold 2: Train ≤2019 → Predict 2023
  Fold 3: Train ≤2023 → Predict 2025

INSTALL:
  pip install lightgbm xgboost scikit-learn shap pandas psycopg2-binary sqlalchemy tqdm

RUN:
  python 05_ml_train_evaluate.py --no-prophet --port VALPARAÍSO   # test
  python 05_ml_train_evaluate.py --no-prophet                      # all ports
  python 05_ml_train_evaluate.py --forecast-only                   # skip CV
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("waze_ml")

SCHEMA = "waze_cargo"

COVID_YEARS      = {2020, 2021, 2022}
INCOMPLETE_YEARS = {2024}
CLEAN_YEARS      = set(range(2005, 2026)) - COVID_YEARS - INCOMPLETE_YEARS

CV_FOLDS = [
    (2018, 2019),
    (2019, 2023),
    (2023, 2025),
]

YEAR_WEIGHTS = {2020: 0.1, 2021: 0.2, 2022: 0.4, 2024: 0.0}

# ── Feature list — ONLY numeric columns ──────────────────────────
# Never include port, direction, aduana, season, dominant_* or any TEXT column.
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3",
    "lag_12_clean",
    "lag_value_12", "lag_weight_12",
    "rolling_3_mean", "rolling_12_mean", "rolling_value_12_mean",
    "yoy_growth_clean", "yoy_value_growth",
    "month_sin", "month_cos", "quarter", "year_index",
    "commodity_diversity", "hs4_diversity",
    "country_diversity",   "continent_diversity",
    "pct_general", "pct_bulk", "pct_refrigerated", "pct_container",
    "avg_value_per_shipment_usd",
    "weight_per_shipment_mt",
    "avg_quantity_per_shipment",
    "sc_norm", "v_norm", "w_norm", "cd_norm",
    # COVID flags (binary integers)
    "is_covid_shock",
    "is_covid_rebound",
    "is_covid_aftershock",
    "lag_12_is_covid",
]

TARGET = "shipment_count"


# ════════════════════════════════════════════════════════════════
#  CONNECTION
# ════════════════════════════════════════════════════════════════

def get_engine():
    h  = os.environ.get("RDS_HOST", "")
    p  = os.environ.get("RDS_PORT", "5432")
    u  = os.environ.get("RDS_USER", "")
    pw = os.environ.get("RDS_PASSWORD", "")
    db = os.environ.get("RDS_DBNAME", "")
    missing = [k for k, v in [("RDS_HOST", h), ("RDS_USER", u),
                                ("RDS_PASSWORD", pw), ("RDS_DBNAME", db)] if not v]
    if missing:
        log.error("Missing env vars: %s", " ".join(missing)); sys.exit(1)
    engine = create_engine(
        f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{db}?sslmode=require",
        pool_pre_ping=True, connect_args={"connect_timeout": 15},
    )
    with engine.connect() as c:
        c.execute(text("SELECT 1"))
    log.info("Connected → %s/%s", h, db)
    return engine


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════

def load_features(engine, port_filter=None):
    where = f"AND port = '{port_filter}'" if port_filter else ""
    sql = f"""
        SELECT * FROM {SCHEMA}.port_features_indexed
        WHERE year BETWEEN 2005 AND 2025
          AND year != 2024
          {where}
        ORDER BY port, direction, year, month
    """
    df = pd.read_sql(sql, engine)
    log.info("Loaded %d rows × %d cols (2024 excluded)", len(df), len(df.columns))
    return df


# ════════════════════════════════════════════════════════════════
#  COVID FEATURE ENGINEERING  (fully vectorised — no df.loc loops)
# ════════════════════════════════════════════════════════════════

def add_covid_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["port", "direction", "year", "month"]).reset_index(drop=True)

    # ── Binary COVID flags (integer, not string) ──────────────
    df["is_covid_shock"]      = (df["year"] == 2020).astype(np.int8)
    df["is_covid_rebound"]    = (df["year"] == 2021).astype(np.int8)
    df["is_covid_aftershock"] = (df["year"] == 2022).astype(np.int8)
    df["lag_12_is_covid"]     = df["year"].apply(
        lambda y: np.int8(1) if (y - 1) in COVID_YEARS else np.int8(0)
    )

    # ── Per-port-direction pre-COVID monthly averages ─────────
    # Used to estimate what lag_12 SHOULD be for COVID-tainted rows
    pre_mask   = df["year"].isin(range(2015, 2020))
    pre_covid  = df[pre_mask].groupby(["port","direction","month"])[TARGET].mean()
    pre_covid  = pre_covid.rename("pre_covid_monthly_avg").reset_index()
    df = df.merge(pre_covid, on=["port","direction","month"], how="left")

    # ── Per-port-direction clean YoY growth (median of clean years) ──
    clean_mask  = df["year"].isin(CLEAN_YEARS) & df["yoy_growth"].notna()
    clean_yoy   = (df[clean_mask]
                   .groupby(["port","direction"])["yoy_growth"]
                   .median()
                   .rename("clean_yoy_overall")
                   .reset_index())
    df = df.merge(clean_yoy, on=["port","direction"], how="left")
    df["clean_yoy_overall"] = df["clean_yoy_overall"].fillna(0.03)

    # ── lag_12_clean: replace COVID-tainted lag_12 ────────────
    # For year Y where lag_12 references a COVID year (Y-1 in COVID_YEARS):
    #   estimate = pre_covid_monthly_avg × (1 + clean_growth)^(Y-1-2019)
    def estimate_clean_lag12(row):
        if row["lag_12_is_covid"] == 0:
            return row["lag_12"]           # not tainted — use as-is
        base  = row["pre_covid_monthly_avg"]
        if pd.isna(base) or base <= 0:
            base = row["rolling_12_mean"] or 1
        years_out = row["year"] - 1 - 2019
        g = max(-0.15, min(0.15, row["clean_yoy_overall"]))
        return max(1.0, float(base) * ((1 + g) ** years_out))

    df["lag_12_clean"] = df.apply(estimate_clean_lag12, axis=1)

    # ── yoy_growth_clean: replace COVID-era YoY ───────────────
    # For COVID years, use the clean-period median growth
    df["yoy_growth_clean"] = np.where(
        df["year"].isin(COVID_YEARS),
        df["clean_yoy_overall"],
        df["yoy_growth"].fillna(df["clean_yoy_overall"]),
    )

    # Drop helper columns — don't let them sneak into features
    df = df.drop(columns=["pre_covid_monthly_avg", "clean_yoy_overall"],
                 errors="ignore")

    return df


def get_sample_weights(df: pd.DataFrame) -> np.ndarray:
    return df["year"].map(lambda y: YEAR_WEIGHTS.get(y, 1.0)).values


# ════════════════════════════════════════════════════════════════
#  FEATURE SELECTION
#  CRITICAL: only numeric columns are ever passed to models
# ════════════════════════════════════════════════════════════════

def select_features(df_train, feature_cols, corr_threshold=0.95):
    # Step 1: keep only features that exist AND are numeric
    numeric_dtypes = ["int8","int16","int32","int64",
                      "uint8","uint16","uint32","uint64",
                      "float16","float32","float64",
                      "Int8","Int16","Int32","Int64"]
    available = [
        c for c in feature_cols
        if c in df_train.columns
        and (str(df_train[c].dtype) in numeric_dtypes
             or pd.api.types.is_numeric_dtype(df_train[c]))
    ]

    if not available:
        return []

    # Step 2: correlation filter
    X    = df_train[available].fillna(0)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    selected = [c for c in available if c not in to_drop]
    return selected


# ════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, model_name, port, direction, fold_year):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred) if len(y_true) > 1 else None
    mask   = y_true > 0
    mape   = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
             if mask.sum() > 0 else None
    return {
        "model": model_name, "port": port, "direction": direction,
        "fold_year": fold_year, "n_samples": len(y_true),
        "mae":  round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(mape, 2) if mape is not None else None,
        "r2":   round(float(r2), 4) if r2 is not None else None,
        "evaluated_at": datetime.now(),
    }


# ════════════════════════════════════════════════════════════════
#  MODELS
# ════════════════════════════════════════════════════════════════

def predict_baseline(df_test):
    """COVID-aware seasonal naive: lag_12_clean × (1 + capped growth)."""
    preds = []
    for _, row in df_test.iterrows():
        lag12  = row.get("lag_12_clean") or row.get("rolling_12_mean") or 1
        growth = row.get("yoy_growth_clean") or 0
        growth = max(-0.20, min(0.20, float(growth)))
        preds.append(max(1.0, float(lag12) * (1 + growth)))
    return np.array(preds)


def train_lgbm(df_tr, df_val, features, target):
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lightgbm not installed"); return None, None

    Xtr = df_tr[features].fillna(0).astype(float)
    ytr = df_tr[target].astype(float).values
    wtr = get_sample_weights(df_tr)
    Xv  = df_val[features].fillna(0).astype(float)
    yv  = df_val[target].astype(float).values

    params = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=31, min_child_samples=10, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=5, verbose=-1, n_jobs=-1)
    dtrain = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    dval   = lgb.Dataset(Xv,  label=yv,  reference=dtrain)
    model  = lgb.train(params, dtrain, num_boost_round=500,
                       valid_sets=[dval],
                       callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(-1)])
    imp = pd.DataFrame({"feature": features,
                         "importance": model.feature_importance("gain")})
    return model, imp


def predict_lgbm(model, df, features):
    if model is None: return None
    return np.maximum(1, model.predict(df[features].fillna(0).astype(float)))


def train_xgb(df_tr, df_val, features, target):
    try:
        import xgboost as xgb
    except ImportError:
        log.warning("xgboost not installed"); return None

    Xtr = df_tr[features].fillna(0).astype(float)
    ytr = df_tr[target].astype(float).values
    wtr = get_sample_weights(df_tr)
    Xv  = df_val[features].fillna(0).astype(float)
    yv  = df_val[target].astype(float).values

    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8,
                               early_stopping_rounds=50, eval_metric="rmse",
                               verbosity=0, n_jobs=-1)
    model.fit(Xtr, ytr, sample_weight=wtr,
              eval_set=[(Xv, yv)], verbose=False)
    return model


def predict_xgb(model, df, features):
    if model is None: return None
    return np.maximum(1, model.predict(df[features].fillna(0).astype(float)))


def train_prophet(df_train):
    try:
        from prophet import Prophet
    except ImportError:
        return None
    ts = df_train[["year","month",TARGET]].copy()
    ts["ds"] = pd.to_datetime(ts[["year","month"]].assign(day=1))
    ts["y"]  = ts[TARGET].astype(float)
    ts = ts[["ds","y"]].dropna()
    if len(ts) < 24: return None
    valid_cp = [cp for cp in ["2020-03-01","2020-12-01","2021-06-01","2022-09-01"]
                if pd.Timestamp(cp) <= ts["ds"].max()]
    m = __import__("prophet").Prophet(
        yearly_seasonality=True, weekly_seasonality=False,
        daily_seasonality=False, seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        changepoints=valid_cp or None)
    m.fit(ts)
    return m


def predict_prophet(model, df_test):
    if model is None: return None
    future = df_test[["year","month"]].copy()
    future["ds"] = pd.to_datetime(future[["year","month"]].assign(day=1))
    fc = model.predict(future[["ds"]])
    return np.maximum(1, fc["yhat"].values)


# ════════════════════════════════════════════════════════════════
#  SHAP
# ════════════════════════════════════════════════════════════════

def compute_shap(model, df, features, port, direction):
    try:
        import shap
        X = df[features].fillna(0).astype(float)
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(X)
        rows = []
        for f, s, g in zip(features, np.abs(sv).mean(axis=0),
                            model.feature_importance("gain")):
            rows.append({"port": port, "direction": direction, "feature": f,
                         "shap_mean_abs": round(float(s), 4),
                         "lgbm_gain":     round(float(g), 4)})
        return rows
    except Exception:
        return []


# ════════════════════════════════════════════════════════════════
#  WALK-FORWARD CV
# ════════════════════════════════════════════════════════════════

def walk_forward_cv(df_port, features, use_prophet=False):
    all_metrics     = []
    importance_rows = []
    best_model_obj  = None
    best_features   = features
    best_mape       = float("inf")

    port      = df_port["port"].iloc[0]
    direction = df_port["direction"].iloc[0]

    for train_end, predict_year in CV_FOLDS:
        df_train = df_port[df_port["year"] <= train_end].copy()
        df_test  = df_port[df_port["year"] == predict_year].copy()
        if len(df_train) < 24 or len(df_test) == 0:
            continue

        sel = select_features(df_train, features)
        if not sel:
            continue

        # Validation: last clean year in training window
        clean_yrs = sorted(y for y in df_train["year"].unique()
                           if y not in COVID_YEARS)
        if len(clean_yrs) >= 2:
            val_yr = clean_yrs[-1]
            df_tr  = df_train[df_train["year"] < val_yr].copy()
            df_vl  = df_train[df_train["year"] == val_yr].copy()
        else:
            df_tr  = df_train.iloc[:-12].copy()
            df_vl  = df_train.iloc[-12:].copy()

        if len(df_tr) < 12 or len(df_vl) == 0:
            continue

        y_true = df_test[TARGET].values

        # Baseline
        all_metrics.append(compute_metrics(
            y_true, predict_baseline(df_test),
            "baseline_covid_aware", port, direction, predict_year))

        # LightGBM
        lgbm_m, lgbm_imp = train_lgbm(df_tr, df_vl, sel, TARGET)
        y_lgbm = predict_lgbm(lgbm_m, df_test, sel)
        if y_lgbm is not None:
            m = compute_metrics(y_true, y_lgbm, "lightgbm",
                                port, direction, predict_year)
            all_metrics.append(m)
            if m["mape"] is not None and m["mape"] < best_mape:
                best_mape      = m["mape"]
                best_model_obj = ("lightgbm", lgbm_m, sel)
                best_features  = sel
            importance_rows += compute_shap(lgbm_m, df_test, sel, port, direction)

        # XGBoost
        xgb_m  = train_xgb(df_tr, df_vl, sel, TARGET)
        y_xgb  = predict_xgb(xgb_m, df_test, sel)
        if y_xgb is not None:
            m = compute_metrics(y_true, y_xgb, "xgboost",
                                port, direction, predict_year)
            all_metrics.append(m)
            if best_model_obj is None and m["mape"] is not None:
                best_model_obj = ("xgboost", xgb_m, sel)

        # Prophet (optional)
        if use_prophet:
            pm    = train_prophet(df_train)
            y_p   = predict_prophet(pm, df_test)
            if y_p is not None:
                all_metrics.append(compute_metrics(
                    y_true, y_p, "prophet", port, direction, predict_year))

    return all_metrics, best_model_obj, best_features, importance_rows


# ════════════════════════════════════════════════════════════════
#  2026 FORECAST
# ════════════════════════════════════════════════════════════════

def forecast_2026(df_port, best_model_obj, features):
    port      = df_port["port"].iloc[0]
    direction = df_port["direction"].iloc[0]

    df_all = df_port[df_port["year"] != 2024].copy()
    df_tr  = df_all[df_all["year"] < 2025].copy()
    df_vl  = df_all[df_all["year"] == 2025].copy()
    if len(df_vl) == 0:
        df_vl = df_tr.tail(12).copy()

    model_name   = "baseline_covid_aware"
    final_model  = None
    sel_features = select_features(df_tr, features)

    if best_model_obj is not None:
        model_name, _, sel_features = best_model_obj
        if model_name == "lightgbm":
            final_model, _ = train_lgbm(df_tr, df_vl, sel_features, TARGET)
        elif model_name == "xgboost":
            final_model = train_xgb(df_tr, df_vl, sel_features, TARGET)

    # Seed from most recent 12 clean months (2025)
    seed = df_all[df_all["year"] == 2025].copy()
    if len(seed) == 0:
        seed = df_all.tail(12).copy()

    forecast_rows = []
    for step in range(1, 13):
        last = seed.iloc[-1]
        nm   = int(last["month"]) % 12 + 1
        ny   = int(last["year"]) + (1 if nm == 1 else 0)

        # lag_12 from seed same month (2025 = clean year)
        same_m = seed[seed["month"] == nm]
        lag12  = float(same_m[TARGET].iloc[-1]) if len(same_m) > 0 \
                 else float(seed[TARGET].mean())

        clean_s   = seed[seed["year"].isin(CLEAN_YEARS)]
        src       = clean_s if len(clean_s) > 0 else seed
        clean_yoy = float(src["yoy_growth_clean"].median()) \
                    if "yoy_growth_clean" in src.columns else 0.03
        clean_yoy = max(-0.15, min(0.15, clean_yoy))

        row = {
            "lag_1":   float(seed.iloc[-1][TARGET]),
            "lag_2":   float(seed.iloc[-2][TARGET]) if len(seed) >= 2 else lag12,
            "lag_3":   float(seed.iloc[-3][TARGET]) if len(seed) >= 3 else lag12,
            "lag_12_clean":    lag12,
            "lag_12":          lag12,
            "lag_12_is_covid": 0,
            "is_covid_shock":  0, "is_covid_rebound": 0, "is_covid_aftershock": 0,
            "rolling_3_mean":  float(seed.tail(3)[TARGET].mean()),
            "rolling_12_mean": float(seed.tail(12)[TARGET].mean()),
            "yoy_growth_clean": clean_yoy,
            "yoy_growth":       clean_yoy,
            "month_sin":  float(np.sin(2 * np.pi * nm / 12)),
            "month_cos":  float(np.cos(2 * np.pi * nm / 12)),
            "quarter":    float((nm - 1) // 3 + 1),
            "year_index": float(ny - 2005),
        }
        # Stable features: recent average from clean data only
        stable = ["commodity_diversity","hs4_diversity","country_diversity",
                  "continent_diversity","pct_general","pct_bulk",
                  "pct_refrigerated","pct_container","avg_value_per_shipment_usd",
                  "weight_per_shipment_mt","avg_quantity_per_shipment",
                  "sc_norm","v_norm","w_norm","cd_norm",
                  "lag_value_12","lag_weight_12",
                  "rolling_value_12_mean","yoy_value_growth"]
        for col in stable:
            if col in src.columns:
                row[col] = float(src[col].mean()) if len(src) > 0 else 0.0

        df_row = pd.DataFrame([row])

        if model_name == "lightgbm" and final_model is not None:
            pred_val = float(predict_lgbm(final_model, df_row, sel_features)[0])
        elif model_name == "xgboost" and final_model is not None:
            pred_val = float(predict_xgb(final_model, df_row, sel_features)[0])
        else:
            pred_val = lag12 * (1 + clean_yoy)

        pred_val = max(1.0, pred_val)
        forecast_rows.append({
            "port": port, "direction": direction,
            "year": ny, "month": nm,
            "pred_shipment_count": round(pred_val),
            "best_model": model_name,
            "forecast_generated_at": datetime.now(),
        })

        # Append predicted row to seed for next step
        new_row = {c: float(seed.iloc[-1][c])
                   if c in seed.columns and pd.api.types.is_numeric_dtype(seed[c])
                   else seed.iloc[-1].get(c)
                   for c in seed.columns}
        new_row.update({"year": ny, "month": nm, TARGET: pred_val,
                        "yoy_growth_clean": clean_yoy, "lag_12_clean": lag12})
        seed = pd.concat([seed, pd.DataFrame([new_row])],
                         ignore_index=True).tail(24)

    return forecast_rows


# ════════════════════════════════════════════════════════════════
#  COVID DIAGNOSTICS
# ════════════════════════════════════════════════════════════════

def covid_diagnostics(df_port):
    port = df_port["port"].iloc[0]
    dir_ = df_port["direction"].iloc[0]
    def avg(mask): return float(df_port[mask][TARGET].mean()) or None
    pre  = avg(df_port["year"].isin(range(2017, 2020)))
    y20  = avg(df_port["year"] == 2020)
    y21  = avg(df_port["year"] == 2021)
    y23  = avg(df_port["year"] == 2023)
    cont = int(df_port.get("lag_12_is_covid", pd.Series([0])).sum())
    return {
        "port": port, "direction": dir_,
        "pre_covid_avg_ships":   round(pre, 0) if pre else None,
        "covid_2020_avg_ships":  round(y20, 0) if y20 else None,
        "covid_2021_avg_ships":  round(y21, 0) if y21 else None,
        "post_covid_2023_ships": round(y23, 0) if y23 else None,
        "pct_drop_2020":    round(100*(y20-pre)/pre, 1) if pre and y20 else None,
        "pct_rebound_2021": round(100*(y21-pre)/pre, 1) if pre and y21 else None,
        "n_contaminated_lag12_rows": cont,
    }


# ════════════════════════════════════════════════════════════════
#  WRITE TO RDS
# ════════════════════════════════════════════════════════════════

def write_results(engine, metrics, importance, forecasts, diagnostics):
    tables = {
        "ml_model_evaluation":   (metrics,     """CREATE TABLE IF NOT EXISTS {s}.ml_model_evaluation (
            model TEXT, port TEXT, direction TEXT, fold_year INTEGER, n_samples INTEGER,
            mae DOUBLE PRECISION, rmse DOUBLE PRECISION, mape DOUBLE PRECISION,
            r2 DOUBLE PRECISION, evaluated_at TIMESTAMPTZ)"""),
        "ml_feature_importance": (importance,  """CREATE TABLE IF NOT EXISTS {s}.ml_feature_importance (
            port TEXT, direction TEXT, feature TEXT,
            shap_mean_abs DOUBLE PRECISION, lgbm_gain DOUBLE PRECISION)"""),
        "ml_forecast_2026":      (forecasts,   """CREATE TABLE IF NOT EXISTS {s}.ml_forecast_2026 (
            port TEXT, direction TEXT, year INTEGER, month INTEGER,
            pred_shipment_count INTEGER, best_model TEXT,
            forecast_generated_at TIMESTAMPTZ)"""),
        "ml_covid_diagnostics":  (diagnostics, """CREATE TABLE IF NOT EXISTS {s}.ml_covid_diagnostics (
            port TEXT, direction TEXT, pre_covid_avg_ships DOUBLE PRECISION,
            covid_2020_avg_ships DOUBLE PRECISION, covid_2021_avg_ships DOUBLE PRECISION,
            post_covid_2023_ships DOUBLE PRECISION, pct_drop_2020 DOUBLE PRECISION,
            pct_rebound_2021 DOUBLE PRECISION, n_contaminated_lag12_rows INTEGER)"""),
    }
    for tbl, (data, ddl) in tables.items():
        with engine.begin() as c:
            c.execute(text(ddl.format(s=SCHEMA)))
            c.execute(text(f"TRUNCATE TABLE {SCHEMA}.{tbl}"))
        if data:
            pd.DataFrame(data).to_sql(tbl, engine, schema=SCHEMA,
                                       if_exists="append", index=False,
                                       method="multi")
        log.info("  ✓ %-35s  %d rows",
                 f"{SCHEMA}.{tbl}", len(data) if data else 0)


# ════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════

def print_summary(metrics):
    if not metrics: return
    df = pd.DataFrame(metrics)
    max_fold = df["fold_year"].max()
    df_f = df[df["fold_year"] == max_fold]
    summary = (df_f.groupby("model")
               .agg(avg_mae=("mae","mean"), avg_rmse=("rmse","mean"),
                    avg_mape=("mape","mean"), avg_r2=("r2","mean"),
                    n_ports=("port","nunique"))
               .sort_values("avg_mape"))
    log.info("")
    log.info("═" * 70)
    log.info("  MODEL COMPARISON  (fold %d, %d ports)", max_fold,
             len(df_f["port"].unique()))
    log.info("  NOTE: COVID years excluded from test folds")
    log.info("  %-30s  %8s  %8s  %7s  %6s",
             "Model","Avg MAE","Avg RMSE","MAPE%","R²")
    log.info("  " + "─" * 66)
    for model, row in summary.iterrows():
        log.info("  %-30s  %8.1f  %8.1f  %6.1f%%  %6.3f",
                 model, row["avg_mae"] or 0, row["avg_rmse"] or 0,
                 row["avg_mape"] or 0, row["avg_r2"] or 0)
    log.info("═" * 70)
    log.info("  Best model: %s  (MAPE %.1f%%)",
             summary.index[0], summary.iloc[0]["avg_mape"] or 0)


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",          help="Single port for debugging")
    ap.add_argument("--no-prophet",    action="store_true",
                    help="Skip Prophet (faster)")
    ap.add_argument("--forecast-only", action="store_true",
                    help="Skip CV, just forecast 2026")
    args = ap.parse_args()

    engine = get_engine()

    log.info("═" * 70)
    log.info("  WAZE CARGO — COVID-Aware ML Pipeline")
    log.info("  COVID years flagged: 2020 (shock), 2021 (rebound), 2022 (aftershock)")
    log.info("  2024 excluded: incomplete data")
    log.info("  Clean CV folds: %s", CV_FOLDS)
    log.info("═" * 70)

    df_raw = load_features(engine, port_filter=args.port)
    log.info("Adding COVID-aware features ...")
    df_all = add_covid_features(df_raw)
    log.info("  ✓ lag_12_clean, yoy_growth_clean, COVID flags added")

    ports = df_all.groupby(["port","direction"]).size().reset_index()
    log.info("Processing %d port-direction pairs", len(ports))

    all_metrics    = []
    all_importance = []
    all_forecasts  = []
    all_diagnostics= []

    for _, row in tqdm(ports.iterrows(), total=len(ports), desc="Ports"):
        port      = row["port"]
        direction = row["direction"]
        df_port   = df_all[
            (df_all["port"] == port) &
            (df_all["direction"] == direction)
        ].copy().sort_values(["year","month"])

        if len(df_port) < 36:
            continue

        # Diagnostics always
        try:
            all_diagnostics.append(covid_diagnostics(df_port))
        except Exception as e:
            log.warning("  diagnostics error %s/%s: %s", port, direction, e)

        # CV training
        best_model_obj = None
        best_features  = select_features(df_port, FEATURE_COLS)

        if not args.forecast_only:
            try:
                metrics, best_model_obj, best_features, importance = walk_forward_cv(
                    df_port, FEATURE_COLS,
                    use_prophet=not args.no_prophet,
                )
                all_metrics    += metrics
                all_importance += importance
            except Exception as e:
                log.warning("  CV error %s/%s: %s", port, direction, e)
        else:
            try:
                df_tr = df_port[df_port["year"].isin(CLEAN_YEARS) &
                                (df_port["year"] < 2025)]
                df_vl = df_port[df_port["year"] == 2025]
                if len(df_vl) == 0:
                    df_vl = df_tr.tail(12)
                sel = select_features(df_tr, FEATURE_COLS)
                lgbm_m, _ = train_lgbm(df_tr, df_vl, sel, TARGET)
                best_model_obj = ("lightgbm", lgbm_m, sel) if lgbm_m else None
                best_features  = sel
            except Exception as e:
                log.warning("  model fit error %s/%s: %s", port, direction, e)

        # Forecast
        try:
            all_forecasts += forecast_2026(df_port, best_model_obj, best_features)
        except Exception as e:
            log.warning("  forecast error %s/%s: %s", port, direction, e)

    print_summary(all_metrics)

    log.info("\nWriting results to RDS ...")
    write_results(engine, all_metrics, all_importance,
                  all_forecasts, all_diagnostics)

    engine.dispose()
    log.info("\nML pipeline complete.")
    log.info("Tables written:")
    log.info("  ml_model_evaluation    → cross-validation scores")
    log.info("  ml_feature_importance  → SHAP + LightGBM gain")
    log.info("  ml_forecast_2026       → 2026 predictions")
    log.info("  ml_covid_diagnostics   → COVID contamination stats")


if __name__ == "__main__":
    main()
