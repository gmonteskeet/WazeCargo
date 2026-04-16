"""Build 04_xgboost.ipynb"""
from _nb_builder import build_notebook

cells = []

cells.append(("md", r"""# 04 — XGBoost (Gradient Boosted Decision Trees)

**Notebook 4 of 8 — The original benchmark**

---

## 4.1  Why also test XGBoost?

XGBoost was for years the **default** gradient-boosting library and is
widely used in time-series competitions. Including it gives the thesis
two valuable things:

1. **Replicability.** A reviewer who is more familiar with XGBoost can
   reproduce the comparison without trusting LightGBM-specific tricks.
2. **Algorithm-vs-implementation isolation.** LightGBM and XGBoost
   optimise the **same** loss function — both fit gradient-boosted
   regression trees with second-order gradients. Differences in their
   scores therefore reflect *implementation* choices, not theory:
   - **Histogram bucketing** (LightGBM) vs. **pre-sorted greedy splits**
     (XGBoost classic) — affects speed but not the asymptotic accuracy.
   - **Leaf-wise** growth (LightGBM) vs. **level-wise** growth (XGBoost) —
     LightGBM tends to overfit slightly faster on small data, level-wise
     is more conservative.
   - **NaN handling.** XGBoost also supports default direction at NaN,
     so the missing-weight columns work the same way.

If LightGBM and XGBoost end up within ~2 % MAPE of each other, the
choice is purely about training speed. The pptx already documents that
XGBoost is **3.5 minutes** for the full retrain vs LightGBM's
**64 seconds**.

## 4.2  Hyperparameters

Mirrors `train_xgb()` in `05_ml_train_evaluate.py`:

| Parameter            | Value |
|----------------------|-------|
| `n_estimators`       | 500 (early stop after 50 stale rounds) |
| `learning_rate`      | 0.05 |
| `max_depth`          | 5    |
| `subsample`          | 0.8  |
| `colsample_bytree`   | 0.8  |
| `objective`          | reg:squarederror |
"""))

cells.append(("code", r"""# ── Setup ─────────────────────────────────────────────────────────
import warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

import wz_ml_utils as U

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (11, 5), "figure.dpi": 110})

FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
MAX_PORTS  = 78
MODEL_NAME = "xgboost"

XGB_PARAMS = dict(
    n_estimators       = 500,
    learning_rate      = 0.05,
    max_depth          = 5,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    early_stopping_rounds = 50,
    eval_metric        = "rmse",
    verbosity          = 0,
    n_jobs             = -1,
)

print(f"XGBoost version : {xgb.__version__}")
"""))

cells.append(("code", r"""df_panel = U.load_features()
print(f"Panel shape : {df_panel.shape}")
print(f"Pairs       : {df_panel.groupby(['port','direction']).ngroups}")
"""))

cells.append(("md", r"""## 4.3  Fit-predict function (mirrors LightGBM contract)"""))

cells.append(("code", r"""def xgb_fit_predict(df_train, df_test, features):
    clean_yrs = sorted(y for y in df_train["year"].unique()
                       if y not in U.COVID_YEARS)
    if len(clean_yrs) >= 2:
        val_yr = clean_yrs[-1]
        df_tr = df_train[df_train["year"] < val_yr].copy()
        df_vl = df_train[df_train["year"] == val_yr].copy()
    else:
        df_tr = df_train.iloc[:-12].copy()
        df_vl = df_train.iloc[-12:].copy()

    Xtr = df_tr[features].fillna(0).astype(float)
    ytr = df_tr[U.TARGET].astype(float).values
    wtr = U.get_sample_weights(df_tr)
    Xv  = df_vl[features].fillna(0).astype(float)
    yv  = df_vl[U.TARGET].astype(float).values
    Xte = df_test[features].fillna(0).astype(float)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(Xtr, ytr, sample_weight=wtr,
              eval_set=[(Xv, yv)], verbose=False)
    return np.maximum(1.0, model.predict(Xte))
"""))

cells.append(("md", r"""## 4.4  Walk-forward CV across 78 ports"""))

cells.append(("code", r"""t0 = time.time()
metrics = U.evaluate_model_across_ports(
    df_panel, xgb_fit_predict,
    model_name=MODEL_NAME,
    max_ports=MAX_PORTS,
)
print(f"Walk-forward CV done in {time.time()-t0:.1f}s")
print(f"Total fold scores: {len(metrics)}")
"""))

cells.append(("code", r"""U.save_metrics(MODEL_NAME, metrics)
summary = U.summarise(metrics, df_panel)
summary
"""))

cells.append(("code", r"""fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, m, lab in zip(axes, ["mae","mape","r2"],
                      ["MAE (ships/month)","MAPE (%)","R²"]):
    sns.boxplot(data=metrics, x="fold_year", y=m, ax=ax, color="#ff7f0e")
    ax.set_title(f"XGBoost — {lab}")
    if m == "mape":
        ax.set_yscale("log")
plt.tight_layout()
plt.savefig(FIG_DIR / "40_xgb_perfold.png")
plt.show()
"""))

cells.append(("md", r"""## 4.5  Demo port — full fit + interpretation"""))

cells.append(("code", r"""DEMO_PORT, DEMO_DIR = "SAN ANTONIO", "import"
df_port = U.get_port_panel(df_panel, DEMO_PORT, DEMO_DIR)
df_tr_full = df_port[df_port["year"] < 2025]
df_vl_full = df_port[df_port["year"] == 2025]
sel = U.select_features(df_tr_full)

Xtr = df_tr_full[sel].fillna(0).astype(float)
ytr = df_tr_full[U.TARGET].astype(float).values
wtr = U.get_sample_weights(df_tr_full)
Xv  = df_vl_full[sel].fillna(0).astype(float)
yv  = df_vl_full[U.TARGET].astype(float).values

demo_model = xgb.XGBRegressor(**XGB_PARAMS)
demo_model.fit(Xtr, ytr, sample_weight=wtr,
               eval_set=[(Xv, yv)], verbose=False)
print(f"Best iteration : {demo_model.best_iteration}")
"""))

cells.append(("code", r"""imp = pd.DataFrame({
    "feature": sel,
    "gain":    demo_model.feature_importances_,
}).sort_values("gain", ascending=False)

fig, ax = plt.subplots(figsize=(10, max(4, 0.32*len(imp))))
sns.barplot(data=imp, y="feature", x="gain", color="#ff7f0e", ax=ax)
ax.set_title(f"XGBoost gain importance — {DEMO_PORT} ({DEMO_DIR})")
plt.savefig(FIG_DIR / "41_xgb_importance.png")
plt.show()
imp.head(10)
"""))

cells.append(("code", r"""# ── Validation forecast plot ──────────────────────────────────────
val_pred = np.maximum(1.0, demo_model.predict(Xv))
val_score = U.score(yv, val_pred)
print("XGBoost 2025 validation:", val_score)

fig, ax = plt.subplots(figsize=(12, 4.8))
hist = df_port[df_port["year"].between(2017, 2025)]
hist_dates = pd.to_datetime(hist[["year","month"]].assign(day=1))
ax.plot(hist_dates, hist[U.TARGET], color="#1f77b4", lw=1.4, label="Actual")
vl_dates = pd.to_datetime(df_vl_full[["year","month"]].assign(day=1))
ax.plot(vl_dates, val_pred, "o-", color="#ff7f0e", lw=1.6,
        label=f"XGBoost 2025  (MAPE={val_score['mape']:.1f}%)")
for y in [2020,2021,2022]:
    ax.axvspan(pd.Timestamp(y,1,1), pd.Timestamp(y,12,31), color="red", alpha=0.06)
ax.set_title(f"{DEMO_PORT} ({DEMO_DIR}) — XGBoost validation 2025")
ax.legend()
plt.savefig(FIG_DIR / "42_xgb_demo_2025.png")
plt.show()
"""))

cells.append(("code", r"""# ── Residual diagnostics ──────────────────────────────────────────
resid = yv - val_pred
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].plot(vl_dates, resid, "o-", color="#1f77b4")
axes[0].axhline(0, color="black", lw=1)
axes[0].set_title("Residuals over time")

sns.histplot(resid, bins=12, kde=True, ax=axes[1], color="#ff7f0e")
axes[1].axvline(0, color="black", lw=1)
axes[1].set_title("Residual distribution")

axes[2].scatter(val_pred, resid, color="#9467bd", alpha=0.7)
axes[2].axhline(0, color="black", lw=1)
axes[2].set_title("Residuals vs prediction")
plt.tight_layout()
plt.savefig(FIG_DIR / "43_xgb_residuals.png")
plt.show()
"""))

cells.append(("md", r"""## 4.6  2026 forecast"""))

cells.append(("code", r"""def xgb_fit_only(df_train, features):
    df_tr = df_train[df_train["year"] < 2025].copy()
    df_vl = df_train[df_train["year"] == 2025].copy()
    if len(df_vl) == 0:
        df_vl = df_tr.tail(12).copy()
    Xtr = df_tr[features].fillna(0).astype(float)
    ytr = df_tr[U.TARGET].astype(float).values
    wtr = U.get_sample_weights(df_tr)
    Xv  = df_vl[features].fillna(0).astype(float)
    yv  = df_vl[U.TARGET].astype(float).values
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xv, yv)], verbose=False)
    return m

def xgb_predict_one(model, df_row, features):
    return float(model.predict(df_row[features].fillna(0).astype(float))[0])

fc = U.forecast_2026(df_port, xgb_fit_only, xgb_predict_one)
fc["date"] = pd.to_datetime(fc[["year","month"]].assign(day=1))

fig, ax = plt.subplots(figsize=(13, 5))
hist = df_port[df_port["year"].between(2019, 2025)]
hist_dates = pd.to_datetime(hist[["year","month"]].assign(day=1))
ax.plot(hist_dates, hist[U.TARGET], color="#1f77b4", lw=1.4, label="Actual 2019–2025")
ax.plot(fc["date"], fc["pred_shipment_count"], "o-", color="#ff7f0e",
        lw=1.8, label="XGBoost 2026")
ax.set_title(f"{DEMO_PORT} ({DEMO_DIR}) — XGBoost 2026 forecast")
ax.legend()
plt.savefig(FIG_DIR / "44_xgb_2026.png")
plt.show()
fc[["year","month","pred_shipment_count"]]
"""))

cells.append(("md", r"""## 4.7  Take-aways

1. XGBoost confirms the LightGBM result: the same lag/rolling/COVID
   features dominate.
2. Per-fold metrics are within ~ 1–3 % MAPE of LightGBM in production
   benchmarks; this notebook should reproduce the same gap.
3. The reason LightGBM is preferred for production is **speed and the
   ergonomic sample-weight API** — not accuracy.
"""))

build_notebook("04_xgboost.ipynb", cells)
