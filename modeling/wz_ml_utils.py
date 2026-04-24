"""
Shared utilities consumed by every model notebook.

Loads the COVID-aware feature panel and exposes:
    - load_features() ........... read parquet, return panel
    - FEATURE_COLS .............. canonical feature list (mirrors production)
    - select_features() ......... drop highly-correlated features
    - get_sample_weights() ...... COVID year down-weighting
    - walk_forward_eval() ....... 3-fold walk-forward CV given a `fit_predict_fn`
    - score() ................... MAE / RMSE / MAPE / R²
    - save_metrics() / load_metrics()  -> tiny json store under data/metrics/
    - aggregate_summary() ....... per-model summary table

Every notebook imports this module so the protocol stays identical.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR    = Path(__file__).parent / "data"
METRICS_DIR = DATA_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

COVID_YEARS      = {2020, 2021, 2022}
INCOMPLETE_YEARS = set()
CLEAN_YEARS      = set(range(2005, 2026)) - COVID_YEARS - INCOMPLETE_YEARS

YEAR_WEIGHTS = {2020: 0.1, 2021: 0.2, 2022: 0.4}

CV_FOLDS = [
    (2018, 2019),     # pre-COVID
    (2019, 2023),     # post-COVID
    (2023, 2024),     # recovery year
    (2024, 2025),     # most recent
]

TARGET = "shipment_count"

# Numeric features used by every notebook.
#
# IMPORTANT — data-leakage exclusions
# -----------------------------------
# Two separate leaks were discovered in the production feature SQL while
# preparing this EDA.  Both are excluded from the ML feature list here:
#
# 1. Min–max normalised columns  (sc_norm, v_norm, w_norm, cd_norm)
#    These are computed with `PARTITION BY port, direction` covering the
#    ENTIRE history of every port, so a row in 2025 already embeds the 2025
#    min/max of the target.  Textbook target leak under walk-forward CV.
#    They were originally engineered as inputs to the congestion-index
#    formula, never as ML inputs.
#
# 2. Inclusive rolling means  (rolling_3_mean, rolling_12_mean,
#    rolling_value_12_mean)
#    The SQL computes these with `ROWS BETWEEN N PRECEDING AND CURRENT ROW`,
#    meaning `rolling_3_mean[t]` contains `shipment_count[t]` itself.
#    A linear model can then reconstruct the target exactly via
#    `shipment_count = 3 * rolling_3_mean - lag_1 - lag_2`, which is why the
#    first run of notebooks showed Ridge/Lasso/ElasticNet hitting R²=1.000.
#    Trees (LightGBM/XGBoost) can only partially exploit this because their
#    binary splits can't represent an exact linear combination, which is why
#    production LightGBM still reports a plausible-looking R²≈0.91.
#    At inference time (forecast_2026) these columns are rebuilt from the
#    PAST 3/12 months only, so the train/inference distributions differ —
#    another reason to drop them from the honest ML comparison.
#
# We deliberately drop both groups so that every model (linear and tree
# alike) only sees information that would have been available at prediction
# time.  See notebook 01 §7 for the full discussion.
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3",
    "lag_12_clean",
    "lag_value_12", "lag_weight_12",
    "yoy_growth_clean", "yoy_value_growth",
    "month_sin", "month_cos", "quarter", "year_index",
    "commodity_diversity", "hs4_diversity",
    "country_diversity",   "continent_diversity",
    "pct_general", "pct_bulk", "pct_refrigerated", "pct_container",
    "avg_value_per_shipment_usd",
    "weight_per_shipment_mt",
    "avg_quantity_per_shipment",
    # Rolling means now use past-only windows (Bug 2 fixed in SQL):
    "rolling_3_mean", "rolling_12_mean", "rolling_value_12_mean",
    # EXCLUDED — full-history min-max normalisations (leak future min/max):
    #     sc_norm, v_norm, w_norm, cd_norm
    "is_covid_shock", "is_covid_rebound", "is_covid_aftershock",
    "lag_12_is_covid",
]


# ── DATA ──────────────────────────────────────────────────────────
def load_features() -> pd.DataFrame:
    """Return the cached COVID-aware panel as a DataFrame."""
    df = pd.read_parquet(DATA_DIR / "port_features_indexed.parquet")
    return df.sort_values(["port", "direction", "year", "month"]).reset_index(drop=True)


def get_port_panel(df: pd.DataFrame, port: str, direction: str) -> pd.DataFrame:
    return (df[(df["port"] == port) & (df["direction"] == direction)]
              .sort_values(["year", "month"])
              .reset_index(drop=True))


def list_eligible_ports(df: pd.DataFrame, min_months: int = 36) -> pd.DataFrame:
    g = df.groupby(["port", "direction"]).agg(
        n=("shipment_count", "size"),
        avg_volume=("shipment_count", "mean"),
    ).reset_index()
    return g[g["n"] >= min_months].sort_values("avg_volume", ascending=False).reset_index(drop=True)


# ── FEATURES ──────────────────────────────────────────────────────
def select_features(df_train: pd.DataFrame,
                    feature_cols=FEATURE_COLS,
                    corr_threshold: float = 0.95) -> list[str]:
    available = [c for c in feature_cols
                 if c in df_train.columns
                 and pd.api.types.is_numeric_dtype(df_train[c])]
    if not available:
        return []
    X = df_train[available].fillna(0)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    return [c for c in available if c not in drop]


def get_sample_weights(df: pd.DataFrame) -> np.ndarray:
    return df["year"].map(lambda y: YEAR_WEIGHTS.get(y, 1.0)).values


# ── METRICS ───────────────────────────────────────────────────────
def score(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    mask = y_true > 0
    mape = (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            if mask.sum() > 0 else np.nan)
    return {"mae": float(mae), "rmse": float(rmse),
            "mape": float(mape), "r2": float(r2),
            "n": int(len(y_true))}


# ── WALK-FORWARD EVAL ─────────────────────────────────────────────
def walk_forward_eval(df_port: pd.DataFrame,
                      fit_predict_fn,
                      feature_cols=FEATURE_COLS) -> list[dict]:
    """
    `fit_predict_fn(df_train, df_test, features) -> y_pred (np.ndarray)`
    Returns one dict per fold (with model name added by the caller).
    """
    out = []
    for train_end, test_year in CV_FOLDS:
        df_tr = df_port[df_port["year"] <= train_end].copy()
        df_te = df_port[df_port["year"] == test_year].copy()
        if len(df_tr) < 24 or len(df_te) == 0:
            continue
        sel = select_features(df_tr, feature_cols)
        if not sel:
            continue
        try:
            y_pred = fit_predict_fn(df_tr, df_te, sel)
        except Exception as e:
            out.append({"fold_year": test_year, "error": str(e)[:120]})
            continue
        m = score(df_te[TARGET].values, y_pred)
        m["fold_year"] = test_year
        m["features_used"] = len(sel)
        out.append(m)
    return out


def evaluate_model_across_ports(df_all: pd.DataFrame,
                                fit_predict_fn,
                                model_name: str,
                                max_ports: int | None = None,
                                feature_cols=FEATURE_COLS) -> pd.DataFrame:
    pairs = list_eligible_ports(df_all)
    if max_ports is not None:
        pairs = pairs.head(max_ports)
    rows = []
    for _, p in pairs.iterrows():
        port, direction = p["port"], p["direction"]
        df_port = get_port_panel(df_all, port, direction)
        for r in walk_forward_eval(df_port, fit_predict_fn, feature_cols):
            r["port"] = port
            r["direction"] = direction
            r["model"] = model_name
            rows.append(r)
    return pd.DataFrame(rows)


# ── METRIC PERSISTENCE  (for the comparison notebook) ─────────────
def save_metrics(model_name: str, metrics_df: pd.DataFrame):
    out = METRICS_DIR / f"{model_name}.parquet"
    metrics_df.to_parquet(out, index=False)
    return out


def load_all_metrics() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(METRICS_DIR.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── PER-MODEL SUMMARY  (volume-weighted) ──────────────────────────
def summarise(metrics_df: pd.DataFrame, df_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-port-fold metrics into per-model averages."""
    if metrics_df.empty:
        return pd.DataFrame()

    weights = (df_panel.groupby(["port", "direction"])[TARGET]
                       .mean().rename("weight").reset_index())
    df = metrics_df.merge(weights, on=["port", "direction"], how="left")
    df["weight"] = df["weight"].fillna(1.0)

    def wavg(x, w):
        x = pd.to_numeric(x, errors="coerce")
        w = pd.to_numeric(w, errors="coerce").fillna(0)
        m = (~x.isna()) & (w > 0)
        if not m.any(): return np.nan
        return float((x[m] * w[m]).sum() / w[m].sum())

    g = df.groupby(["model", "fold_year"], dropna=False)
    rows = []
    for (model, fold), sub in g:
        rows.append({
            "model":   model,
            "fold_year": fold,
            "n_ports":   sub["port"].nunique(),
            "wmae":  wavg(sub["mae"],  sub["weight"]),
            "wrmse": wavg(sub["rmse"], sub["weight"]),
            "wmape": wavg(sub["mape"], sub["weight"]),
            "wr2":   wavg(sub["r2"],   sub["weight"]),
        })
    return pd.DataFrame(rows)


# ── 12-STEP RECURSIVE FORECAST FOR 2026 ───────────────────────────
def forecast_2026(df_port: pd.DataFrame,
                  fit_fn,
                  predict_fn,
                  feature_cols=FEATURE_COLS) -> pd.DataFrame:
    """
    Train on all clean rows ≤ 2025, recursively predict 12 monthly steps for 2026.

    fit_fn(df_train, features) -> trained_model
    predict_fn(model, df_row, features) -> float
    """
    df_all = df_port.copy()
    df_tr  = df_all[df_all["year"] <= 2025].copy()

    sel = select_features(df_tr, feature_cols)
    if not sel:
        return pd.DataFrame()

    model = fit_fn(df_tr, sel)
    if model is None:
        return pd.DataFrame()

    seed = df_all[df_all["year"] == 2025].copy()
    if len(seed) == 0:
        seed = df_all.tail(12).copy()

    rows = []
    for step in range(1, 13):
        last = seed.iloc[-1]
        nm = int(last["month"]) % 12 + 1
        ny = int(last["year"]) + (1 if nm == 1 else 0)

        same_m = seed[seed["month"] == nm]
        lag12  = float(same_m[TARGET].iloc[-1]) if len(same_m) > 0 \
                 else float(seed[TARGET].mean())

        clean = seed[seed["year"].isin(CLEAN_YEARS)]
        src = clean if len(clean) else seed
        clean_yoy = max(-0.15, min(0.15,
            float(src["yoy_growth_clean"].median())
            if "yoy_growth_clean" in src.columns else 0.03))

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
        stable = ["commodity_diversity", "hs4_diversity", "country_diversity",
                  "continent_diversity", "pct_general", "pct_bulk",
                  "pct_refrigerated", "pct_container", "avg_value_per_shipment_usd",
                  "weight_per_shipment_mt", "avg_quantity_per_shipment",
                  "sc_norm", "v_norm", "w_norm", "cd_norm",
                  "lag_value_12", "lag_weight_12",
                  "rolling_value_12_mean", "yoy_value_growth"]
        for c in stable:
            if c in src.columns:
                row[c] = float(src[c].mean()) if len(src) > 0 else 0.0

        df_row = pd.DataFrame([row])
        for c in sel:
            if c not in df_row.columns:
                df_row[c] = 0.0

        pred = max(1.0, float(predict_fn(model, df_row, sel)))

        rows.append({"year": ny, "month": nm, "pred_shipment_count": pred})

        new_row = {c: (float(seed.iloc[-1][c])
                       if c in seed.columns and pd.api.types.is_numeric_dtype(seed[c])
                       else seed.iloc[-1].get(c))
                   for c in seed.columns}
        new_row.update({"year": ny, "month": nm, TARGET: pred,
                        "yoy_growth_clean": clean_yoy, "lag_12_clean": lag12})
        seed = pd.concat([seed, pd.DataFrame([new_row])],
                          ignore_index=True).tail(24)

    return pd.DataFrame(rows)
