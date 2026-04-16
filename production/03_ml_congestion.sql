-- ══════════════════════════════════════════════════════════════════
-- WAZE CARGO — Port Congestion Forecasting ML
-- Run directly on AWS RDS PostgreSQL
--
-- Usage:
--   psql "host=HOST user=waze_admin dbname=waze_cargo sslmode=require" \
--        -f 03_ml_congestion.sql
--
-- Builds 5 tables:
--   port_monthly_agg          → monthly aggregates per port
--   port_features_indexed     → lag features + congestion index
--   port_seasonal_index       → seasonal factors
--   port_forecast_params      → per-port trend params
--   port_congestion_forecast  → 12-month forecast
-- ══════════════════════════════════════════════════════════════════

SET search_path TO waze_cargo, public;
\echo '════════════════════════════════════════════════════════'
\echo ' WAZE CARGO — Congestion ML'
\echo '════════════════════════════════════════════════════════'

-- ── 1. MONTHLY AGGREGATION ────────────────────────────────────────
\echo ''
\echo 'Step 1/5 — Building port_monthly_agg ...'

DROP TABLE IF EXISTS port_monthly_agg CASCADE;
CREATE TABLE port_monthly_agg AS

SELECT
    i.periodo::INTEGER                                                  AS year,
    i.mes::INTEGER                                                      AS month,
    i.puerto_desembarque                                                AS port,
    'import'::TEXT                                                      AS direction,
    i.aduana,
    COUNT(*)::BIGINT                                                    AS shipment_count,
    COALESCE(SUM(i.cif_us), 0)                                         AS total_value_usd,
    NULL::DOUBLE PRECISION                                              AS total_weight_mt,
    COALESCE(SUM(i.cantidad_mercancia), 0)                             AS total_quantity,
    COUNT(DISTINCT i.hs2_capitulo)::BIGINT                             AS commodity_diversity,
    COUNT(DISTINCT i.hs4_partida)::BIGINT                              AS hs4_diversity,
    COUNT(DISTINCT i.pais_origen)::BIGINT                              AS country_diversity,
    COUNT(DISTINCT i.continente_origen)::BIGINT                        AS continent_diversity,
    MODE() WITHIN GROUP (ORDER BY i.hs2_capitulo)                      AS dominant_hs2,
    MODE() WITHIN GROUP (ORDER BY i.tipo_carga)                        AS dominant_cargo_type,
    MODE() WITHIN GROUP (ORDER BY i.pais_origen)                       AS dominant_origin_country,
    MODE() WITHIN GROUP (ORDER BY i.continente_origen)                 AS dominant_continent,
    SUM(CASE WHEN i.tipo_carga ILIKE '%GENERAL%'    THEN 1 ELSE 0 END)::BIGINT  AS cnt_general,
    SUM(CASE WHEN i.tipo_carga ILIKE '%GRANEL%'     THEN 1 ELSE 0 END)::BIGINT  AS cnt_bulk,
    SUM(CASE WHEN i.tipo_carga ILIKE '%FRIGORI%'    THEN 1 ELSE 0 END)::BIGINT  AS cnt_refrigerated,
    SUM(CASE WHEN i.tipo_carga ILIKE '%CONTENEDOR%' THEN 1 ELSE 0 END)::BIGINT  AS cnt_container,
    AVG(i.cif_us)                                                      AS avg_value_per_shipment_usd
FROM clean_maritime_imports i
WHERE i.puerto_desembarque IS NOT NULL
  AND i.periodo BETWEEN 2005 AND 2025
GROUP BY i.puerto_desembarque, i.periodo, i.mes, i.aduana

UNION ALL

SELECT
    e.periodo::INTEGER,
    e.mes::INTEGER,
    e.puerto_embarque                                                   AS port,
    'export'::TEXT                                                      AS direction,
    e.aduana,
    COUNT(*)::BIGINT,
    COALESCE(SUM(e.fob_us), 0),
    COALESCE(SUM(e.peso_bruto_kg) / 1000.0, 0)                        AS total_weight_mt,
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
FROM clean_maritime_exports e
WHERE e.puerto_embarque IS NOT NULL
  AND e.periodo BETWEEN 2005 AND 2025
GROUP BY e.puerto_embarque, e.periodo, e.mes, e.aduana;

CREATE INDEX ON port_monthly_agg (port, direction, year, month);

SELECT 'port_monthly_agg: ' || COUNT(*) || ' rows' AS status FROM port_monthly_agg;


-- ── 2. FEATURES + CONGESTION INDEX ───────────────────────────────
\echo 'Step 2/5 — Building port_features_indexed ...'

DROP TABLE IF EXISTS port_features_indexed CASCADE;
CREATE TABLE port_features_indexed AS
WITH base AS (
    SELECT *,
        CASE
            WHEN month IN (12,1,2) THEN 'Summer'
            WHEN month IN (3,4,5)  THEN 'Autumn'
            WHEN month IN (6,7,8)  THEN 'Winter'
            ELSE 'Spring'
        END                                                              AS season,
        ((month - 1) / 3 + 1)::DOUBLE PRECISION                         AS quarter,
        (year - 2005)::INTEGER                                           AS year_index,
        SIN(2 * PI() * month / 12.0)                                    AS month_sin,
        COS(2 * PI() * month / 12.0)                                    AS month_cos,
        cnt_general      * 1.0 / NULLIF(shipment_count, 0)              AS pct_general,
        cnt_bulk         * 1.0 / NULLIF(shipment_count, 0)              AS pct_bulk,
        cnt_refrigerated * 1.0 / NULLIF(shipment_count, 0)              AS pct_refrigerated,
        cnt_container    * 1.0 / NULLIF(shipment_count, 0)              AS pct_container,
        COALESCE(total_weight_mt, 0) / NULLIF(shipment_count, 0)        AS weight_per_shipment_mt,
        total_quantity               / NULLIF(shipment_count, 0)        AS avg_quantity_per_shipment,
        LAG(shipment_count,  1) OVER w                                   AS lag_1,
        LAG(shipment_count,  2) OVER w                                   AS lag_2,
        LAG(shipment_count,  3) OVER w                                   AS lag_3,
        LAG(shipment_count, 12) OVER w                                   AS lag_12,
        LAG(total_value_usd, 12) OVER w                                  AS lag_value_12,
        LAG(total_weight_mt, 12) OVER w                                  AS lag_weight_12,
        AVG(shipment_count::DOUBLE PRECISION)
            OVER (PARTITION BY port, direction ORDER BY year, month
                  ROWS BETWEEN 3  PRECEDING AND 1 PRECEDING)           AS rolling_3_mean,
        AVG(shipment_count::DOUBLE PRECISION)
            OVER (PARTITION BY port, direction ORDER BY year, month
                  ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)           AS rolling_12_mean,
        AVG(total_value_usd)
            OVER (PARTITION BY port, direction ORDER BY year, month
                  ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING)           AS rolling_value_12_mean,
        (shipment_count - LAG(shipment_count, 12) OVER w)::DOUBLE PRECISION
            / NULLIF(LAG(shipment_count, 12) OVER w, 0)                 AS yoy_growth,
        (total_value_usd - LAG(total_value_usd, 12) OVER w)
            / NULLIF(LAG(total_value_usd, 12) OVER w, 0)                AS yoy_value_growth
    FROM port_monthly_agg
    WINDOW w AS (PARTITION BY port, direction ORDER BY year, month)
),
filtered AS (
    SELECT * FROM base WHERE lag_1 IS NOT NULL AND lag_12 IS NOT NULL
),
with_norm AS (
    SELECT *,
        (shipment_count  - MIN(shipment_count)  OVER pd)
            / NULLIF(MAX(shipment_count)  OVER pd - MIN(shipment_count)  OVER pd, 0) AS sc_norm,
        (total_value_usd - MIN(total_value_usd) OVER pd)
            / NULLIF(MAX(total_value_usd) OVER pd - MIN(total_value_usd) OVER pd, 0) AS v_norm,
        (COALESCE(total_weight_mt,0) - MIN(COALESCE(total_weight_mt,0)) OVER pd)
            / NULLIF(MAX(COALESCE(total_weight_mt,0)) OVER pd
                   - MIN(COALESCE(total_weight_mt,0)) OVER pd, 0)                    AS w_norm,
        (commodity_diversity - MIN(commodity_diversity) OVER pd)
            / NULLIF(MAX(commodity_diversity) OVER pd
                   - MIN(commodity_diversity) OVER pd, 0)                            AS cd_norm
    FROM filtered
    WINDOW pd AS (PARTITION BY port, direction)
)
SELECT *,
    CASE direction
        WHEN 'import' THEN
            COALESCE(sc_norm,0)*0.40 + COALESCE(v_norm,0)*0.30
            + COALESCE(cd_norm,0)*0.20 + COALESCE(pct_container,0)*0.10
        ELSE
            COALESCE(w_norm,0)*0.40  + COALESCE(sc_norm,0)*0.30
            + COALESCE(v_norm,0)*0.20 + COALESCE(pct_refrigerated,0)*0.10
    END AS congestion_index
FROM with_norm;

CREATE INDEX ON port_features_indexed (port, direction, year, month);

SELECT 'port_features_indexed: ' || COUNT(*) || ' rows' AS status FROM port_features_indexed;


-- ── 3. SEASONAL INDEX ─────────────────────────────────────────────
\echo 'Step 3/5 — Building port_seasonal_index ...'

DROP TABLE IF EXISTS port_seasonal_index CASCADE;
CREATE TABLE port_seasonal_index AS
SELECT
    port, direction, month,
    AVG(shipment_count)::DOUBLE PRECISION
        / NULLIF(AVG(AVG(shipment_count::DOUBLE PRECISION))
                 OVER (PARTITION BY port, direction), 0)        AS seasonal_factor,
    AVG(COALESCE(total_weight_mt, 0))
        / NULLIF(AVG(AVG(COALESCE(total_weight_mt, 0)))
                 OVER (PARTITION BY port, direction), 0)        AS seasonal_weight_factor,
    AVG(congestion_index)                                       AS hist_avg_congestion
FROM port_features_indexed
WHERE year >= 2015
GROUP BY port, direction, month;

SELECT 'port_seasonal_index: ' || COUNT(*) || ' rows' AS status FROM port_seasonal_index;


-- ── 4. FORECAST PARAMS ────────────────────────────────────────────
\echo 'Step 4/5 — Building port_forecast_params ...'

DROP TABLE IF EXISTS port_forecast_params CASCADE;
CREATE TABLE port_forecast_params AS
WITH last_p AS (
    SELECT port, direction,
        (MAX(year * 100 + month) / 100)::INTEGER AS last_year,
        (MAX(year * 100 + month) % 100)::INTEGER AS last_month
    FROM port_features_indexed
    GROUP BY port, direction
)
SELECT
    f.port, f.direction,
    l.last_year, l.last_month,
    AVG(CASE WHEN f.yoy_growth IS NOT NULL AND f.lag_12 > 0
             THEN f.yoy_growth END)                             AS avg_yoy_growth,
    AVG(f.yoy_value_growth)                                    AS avg_yoy_value_growth,
    AVG(f.shipment_count::DOUBLE PRECISION)                    AS avg_shipments,
    AVG(f.total_value_usd)                                     AS avg_value_usd,
    AVG(COALESCE(f.total_weight_mt, 0))                        AS avg_weight_mt,
    AVG(f.commodity_diversity::DOUBLE PRECISION)               AS avg_commodity_diversity,
    AVG(f.hs4_diversity::DOUBLE PRECISION)                     AS avg_hs4_diversity,
    AVG(f.country_diversity::DOUBLE PRECISION)                 AS avg_country_diversity,
    AVG(f.pct_general)                                         AS avg_pct_general,
    AVG(f.pct_bulk)                                            AS avg_pct_bulk,
    AVG(f.pct_refrigerated)                                    AS avg_pct_refrigerated,
    AVG(f.pct_container)                                       AS avg_pct_container,
    AVG(f.weight_per_shipment_mt)                              AS avg_weight_per_shipment_mt,
    AVG(f.avg_quantity_per_shipment)                           AS avg_qty_per_shipment,
    MODE() WITHIN GROUP (ORDER BY f.dominant_hs2)              AS dominant_hs2,
    MODE() WITHIN GROUP (ORDER BY f.dominant_cargo_type)       AS dominant_cargo_type,
    MODE() WITHIN GROUP (ORDER BY f.dominant_continent)        AS dominant_continent,
    COUNT(*)::BIGINT                                           AS n_obs
FROM port_features_indexed f
JOIN last_p l ON f.port = l.port AND f.direction = l.direction
WHERE f.year >= 2018
GROUP BY f.port, f.direction, l.last_year, l.last_month
HAVING COUNT(*) >= 24;

SELECT 'port_forecast_params: ' || COUNT(*) || ' rows' AS status FROM port_forecast_params;


-- ── 5. CONGESTION FORECAST ────────────────────────────────────────
\echo 'Step 5/5 — Building port_congestion_forecast ...'

DROP TABLE IF EXISTS port_congestion_forecast CASCADE;
CREATE TABLE port_congestion_forecast AS
WITH offsets AS (SELECT generate_series(1, 12) AS step),
grid AS (
    SELECT p.*, o.step,
        CASE WHEN (p.last_month + o.step) > 12
             THEN p.last_year + ((p.last_month + o.step - 1) / 12)
             ELSE p.last_year
        END::INTEGER AS forecast_year,
        (((p.last_month + o.step - 1) % 12) + 1)::INTEGER AS forecast_month
    FROM port_forecast_params p CROSS JOIN offsets o
),
with_s AS (
    SELECT g.*,
        LEAST(3.0, GREATEST(0.1, COALESCE(s.seasonal_factor,        1.0))) AS sfob,
        LEAST(3.0, GREATEST(0.1, COALESCE(s.seasonal_weight_factor, 1.0))) AS swt,
        COALESCE(s.hist_avg_congestion, 0.35)                              AS base_cong
    FROM grid g
    LEFT JOIN port_seasonal_index s
           ON s.port      = g.port
          AND s.direction = g.direction
          AND s.month     = g.forecast_month
),
computed AS (
    SELECT *,
        POWER(1 + GREATEST(-0.30, LEAST(0.40,
              COALESCE(NULLIF(avg_yoy_growth,      0), 0.025))), step/12.0) AS gf,
        POWER(1 + GREATEST(-0.30, LEAST(0.40,
              COALESCE(NULLIF(avg_yoy_value_growth, 0), 0.025))), step/12.0) AS gv,
        LEAST(1.0, GREATEST(0.0,
            base_cong
            * LEAST(1.5, GREATEST(0.5,
                LEAST(3.0, GREATEST(0.1, COALESCE(
                    (SELECT seasonal_factor FROM port_seasonal_index s2
                     WHERE s2.port=port AND s2.direction=direction
                       AND s2.month=forecast_month), 1.0)))) )
            * LEAST(1.1, GREATEST(0.95,
                POWER(1 + GREATEST(-0.30, LEAST(0.40,
                      COALESCE(NULLIF(avg_yoy_growth, 0), 0.025))),
                      step/12.0)))
        )) AS ci
    FROM with_s
)
SELECT
    port,
    direction,
    forecast_year                                                       AS year,
    forecast_month                                                      AS month,
    CASE WHEN forecast_month IN (12,1,2) THEN 'Summer'
         WHEN forecast_month IN (3,4,5)  THEN 'Autumn'
         WHEN forecast_month IN (6,7,8)  THEN 'Winter'
         ELSE 'Spring' END                                              AS season,
    GREATEST(1, ROUND(avg_shipments * sfob * gf))::INTEGER              AS pred_shipment_count,
    CASE WHEN direction = 'export'
         THEN ROUND((avg_weight_mt * swt * gv)::NUMERIC, 2)::DOUBLE PRECISION
    END                                                                 AS pred_weight_mt,
    ROUND((avg_qty_per_shipment * avg_shipments * sfob * gf)::NUMERIC,2)::DOUBLE PRECISION
                                                                        AS pred_total_quantity,
    ROUND((avg_value_usd * sfob * gv)::NUMERIC, 2)::DOUBLE PRECISION   AS pred_value_usd,
    ci                                                                  AS congestion_index,
    CASE WHEN ci < 0.25 THEN 'Low'
         WHEN ci < 0.50 THEN 'Moderate'
         WHEN ci < 0.75 THEN 'High'
         ELSE 'Critical' END                                            AS congestion_label,
    ROUND(avg_commodity_diversity::NUMERIC, 1)::DOUBLE PRECISION        AS avg_commodity_diversity,
    ROUND(avg_hs4_diversity::NUMERIC, 1)::DOUBLE PRECISION              AS avg_hs4_diversity,
    ROUND(avg_country_diversity::NUMERIC, 1)::DOUBLE PRECISION          AS avg_country_diversity,
    ROUND(avg_pct_general::NUMERIC, 4)::DOUBLE PRECISION                AS pct_general,
    ROUND(avg_pct_bulk::NUMERIC, 4)::DOUBLE PRECISION                   AS pct_bulk,
    ROUND(avg_pct_refrigerated::NUMERIC, 4)::DOUBLE PRECISION           AS pct_refrigerated,
    ROUND(avg_pct_container::NUMERIC, 4)::DOUBLE PRECISION              AS pct_container,
    dominant_hs2                                                        AS dominant_commodity_hs2,
    dominant_cargo_type,
    dominant_continent,
    'v2.0-rds'                                                          AS model_version,
    NOW()                                                               AS forecast_generated_at
FROM computed
ORDER BY port, direction, year, month;

SELECT 'port_congestion_forecast: ' || COUNT(*) || ' rows' AS status
FROM port_congestion_forecast;

\echo ''
\echo '── Congestion label distribution ────────────────────────'
SELECT congestion_label, COUNT(*) AS n
FROM port_congestion_forecast
GROUP BY 1 ORDER BY 2 DESC;

\echo ''
\echo '════════════════════════════════════════════════════════'
\echo ' Congestion ML COMPLETE'
\echo '════════════════════════════════════════════════════════'
