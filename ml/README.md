┌────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE                                 │
│                                                                │
│  1. DATA SPLIT                                                 │
│     Train: 2005–2022  (17 years)                               │
│     Validation: 2023  (tune hyperparams)                       │
│     Test: 2024–2025   (never touched during training)          │
│                                                                │
│  2. MODELS TO COMPARE                                          │
│     Baseline      → Naïve seasonal (what we have now)          │
│     LightGBM      → gradient boosting on lag features          │
│     XGBoost       → same, different regularisation             │
│     Prophet       → Facebook time series (trend + seasonality) │
│     LSTM          → if you want deep learning                  │
│                                                                │
│  3. FEATURE SELECTION                                          │
│     Correlation filter  → drop features correlated > 0.95      │
│     SHAP values         → which features actually matter       │
│     Recursive feature   → backward elimination by CV score     │
│     elimination (RFE)                                          │
│                                                                │
│  4. EVALUATION METRICS per port                                │
│     MAE   → mean absolute error (shipments)                    │
│     RMSE  → penalises big misses                               │
│     MAPE  → % error (comparable across ports)                  │
│     R²    → variance explained                                 │
│                                                                │
│  5. RETRAIN TRIGGER                                            │
│     When new year added → re-split → retrain → compare         │
│     If new model beats baseline MAPE → promote to production   │
└────────────────────────────────────────────────────────────────┘

a) 05_ml_train_evaluate.py

The COVID problem in data — what we found
YearShipmentsChangeProblem it causes2019981K+5%Clean baseline2020875K-11%lag_12 in 2021 = this → artificially low20211,018K+16%yoy_growth = +50–300% (comparing to COVID low)2022967K-5%lag_12 = 2021 rebound → artificially high2023908K-6%Clean again2024180K-80%Incomplete year — only partial data loaded20251,810K+908%lag_12 = 2024 incomplete → meaningless
The specific contamination: lag_12 in 2021 references COVID-suppressed 2020 values — so the model would "learn" that when lag_12 is low, a +50% jump is normal. That's not a pattern, it's an anomaly.

Seven fixes applied

lag_12_clean — for 2021 and 2022, replaces the COVID-contaminated lag_12 with an interpolated estimate based on the pre-COVID 2015–2019 trend
yoy_growth_clean — replaces the meaningless 2020–2022 YoY growth rates with the median clean-year growth for that port+month
COVID flag features — is_covid_shock, is_covid_rebound, is_covid_aftershock, lag_12_is_covid — explicit binary signals so the model knows these observations are anomalous
Sample weights — 2020 gets weight 0.1, 2021 gets 0.2, 2022 gets 0.4 in LightGBM/XGBoost training. 2024 gets 0.0 (excluded entirely)
Clean CV folds only — folds never use 2020/2021/2022 as test years. Fold 2 jumps from predicting 2019 straight to predicting 2023
2024 excluded from load — the load_features() query has AND year != 2024 because the data is incomplete
Prophet changepoints — explicitly tells Prophet when the structural breaks happened (2020-03, 2020-12, 2021-06, 2022-09) so it doesn't confuse COVID as "normal seasonality"

What the real pipeline does

Data split (walk-forward, never leaks future into past):
Fold 1:  Train 2005–2019  →  Predict 2020  →  Measure error
Fold 2:  Train 2005–2020  →  Predict 2021  →  Measure error
Fold 3:  Train 2005–2021  →  Predict 2022  →  Measure error
Fold 4:  Train 2005–2022  →  Predict 2023  →  Final validation
Final:   Train 2005–2025  →  Forecast 2026  (the output you want)
Feature selection:

Correlation filter removes redundant features (>0.95 correlated)
LightGBM gain importance ranks which features actually help
SHAP values explain why each feature matters per port

Four models compete, worst MAPE loses:
ModelStrengthSeasonal naiveBaseline — anything worse than this is uselessLightGBMBest for tabular time series with lag featuresXGBoostSimilar, different regularisationProphetHandles trend breaks and holiday effects
When you add a new year of data — just re-run the script. The walk-forward folds automatically extend, the models retrain, and the best model for each port gets promoted to the 2026 forecast.