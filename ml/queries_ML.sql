SET search_path TO waze_cargo;

-- 1. Model performance summary
SELECT model,
       COUNT(DISTINCT port)       AS ports,
       ROUND(AVG(mae)::NUMERIC,0) AS avg_mae,
       ROUND(AVG(rmse)::NUMERIC,0)AS avg_rmse,
       ROUND(AVG(r2)::NUMERIC,3)  AS avg_r2
FROM ml_model_evaluation
WHERE fold_year = 2025
GROUP BY model ORDER BY avg_r2 DESC;

-- 2. Which model won per port (best MAPE in last fold)
SELECT DISTINCT ON (port, direction)
       port, direction, model AS best_model,
       ROUND(mape::NUMERIC,1) AS mape_pct,
       ROUND(r2::NUMERIC,3)   AS r2
FROM ml_model_evaluation
WHERE fold_year = 2025
ORDER BY port, direction, mape;

-- 3. Top 10 forecast peaks for 2026
SELECT port, direction, year, month,
       pred_shipment_count, best_model
FROM ml_forecast_2026
ORDER BY pred_shipment_count DESC
LIMIT 10;

-- 4. COVID contamination — worst-affected ports
SELECT port, direction,
       pct_drop_2020,
       pct_rebound_2021,
       n_contaminated_lag12_rows
FROM ml_covid_diagnostics
ORDER BY ABS(pct_drop_2020) DESC NULLS LAST
LIMIT 10;

-- 5. Top SHAP features across all ports
SELECT feature,
       ROUND(AVG(shap_mean_abs)::NUMERIC,4) AS avg_shap,
       ROUND(AVG(lgbm_gain)::NUMERIC,0)     AS avg_gain,
       COUNT(DISTINCT port)                  AS n_ports
FROM ml_feature_importance
GROUP BY feature
ORDER BY avg_shap DESC
LIMIT 15;