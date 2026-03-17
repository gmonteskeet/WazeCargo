-- ══════════════════════════════════════════════════════════════════
-- WAZE CARGO — AWS RDS PostgreSQL Schema
-- Run ONCE on a fresh RDS instance before any data migration.
--
-- Usage:
--   export PGPASSWORD='your_password'
--   psql "host=wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com \
--         user=waze_admin dbname=waze_cargo sslmode=require" \
--        -f 01_rds_schema.sql
--
-- Creates:
--   · Schema: waze_cargo
--   · 13 lookup tables
--   · clean_maritime_imports   (~15.5M rows)
--   · clean_maritime_exports   (~4.5M rows)
--   · 5 ML aggregation tables  (port_monthly_agg, port_features_indexed,
--                                port_seasonal_index, port_forecast_params,
--                                port_congestion_forecast)
--   · 3 commodity ML tables    (commodity_port_params,
--                                commodity_seasonal_index,
--                                commodity_port_forecast)
--   · 4 ML pipeline output tables (ml_model_evaluation,
--                                   ml_feature_importance,
--                                   ml_forecast_2026,
--                                   ml_covid_diagnostics)
--
-- DESIGN NOTES:
--   · ALL text columns use TEXT (no VARCHAR length limit)
--     → avoids StringDataRightTruncation errors seen in migration
--   · periodo and mes have NO NOT NULL constraint
--     → source DuckDB has 754K rows where these are legitimately NULL
--   · HUGEINT from DuckDB maps to BIGINT in PostgreSQL
--   · id on core tables uses BIGSERIAL (auto-increment, not copied from DuckDB)
-- ══════════════════════════════════════════════════════════════════

-- ── Schema ────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS waze_cargo;
SET search_path TO waze_cargo, public;

-- ══════════════════════════════════════════════════════════════════
--  LOOKUP TABLES  (13 tables, small — loaded first)
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS lkp_aduanas (
    cod_aduana_tramitacion  INTEGER PRIMARY KEY,
    nombre_aduana           TEXT
);

CREATE TABLE IF NOT EXISTS lkp_clausulas (
    cl_compra               INTEGER PRIMARY KEY,
    nombre_clausula         TEXT,
    sigla_clausula          TEXT       -- max observed: 62 chars
);

CREATE TABLE IF NOT EXISTS lkp_harmonized_system (
    hscode                  TEXT PRIMARY KEY,
    section                 TEXT,
    description             TEXT,      -- max observed: 255 chars
    parent                  TEXT,
    level                   TEXT
);

CREATE TABLE IF NOT EXISTS lkp_modalidades_venta (
    cod_modalidad_venta         INTEGER PRIMARY KEY,
    nombre_modalidad_venta      TEXT,
    descripcion_modalidad_venta TEXT    -- max observed: 1037 chars
);

CREATE TABLE IF NOT EXISTS lkp_moneda (
    moneda      INTEGER PRIMARY KEY,
    moneda_1    TEXT,
    pais_moneda TEXT
);

CREATE TABLE IF NOT EXISTS lkp_paises (
    cod_pais            INTEGER PRIMARY KEY,
    nombre_pais         TEXT,          -- max observed: 125 chars
    nombre_continente   TEXT
);

CREATE TABLE IF NOT EXISTS lkp_puertos (
    cod_puerto      INTEGER PRIMARY KEY,
    nombre_puerto   TEXT,              -- max observed: 93 chars
    tipo_puerto     TEXT,
    cod_pais        DOUBLE PRECISION,
    pais            TEXT,
    zona_geografica TEXT
);

CREATE TABLE IF NOT EXISTS lkp_regimen_importacion (
    cod_rgimen_importacion      INTEGER PRIMARY KEY,
    nombre_rgimen_importacion   TEXT,  -- max observed: 114 chars
    sigla_rgimen_importacion    TEXT
);

CREATE TABLE IF NOT EXISTS lkp_regiones (
    cod_region_origen   INTEGER PRIMARY KEY,
    nombre_region       TEXT
);

CREATE TABLE IF NOT EXISTS lkp_tipos_carga (
    cod_tipo_carga          TEXT PRIMARY KEY,
    nombre_tipo_carga       TEXT,
    descripcion_tipo_carga  TEXT       -- max observed: 226 chars
);

CREATE TABLE IF NOT EXISTS lkp_tipos_operacion (
    cod_tipo_operacion      INTEGER PRIMARY KEY,
    nombre_tipo_operacion   TEXT,      -- max observed: 59 chars
    nombre_a_consignar      TEXT,
    ingreso_salida          TEXT,      -- max observed: 7 chars
    operacion               TEXT       -- max observed: 33 chars
);

CREATE TABLE IF NOT EXISTS lkp_unidades_medida (
    cod_unidad_medida   INTEGER PRIMARY KEY,
    unidad_medida       TEXT,
    nombre_unidad_medida TEXT          -- max observed: 49 chars
);

CREATE TABLE IF NOT EXISTS lkp_vias_transporte (
    cod_via_transporte      INTEGER PRIMARY KEY,
    nombre_via_transporte   TEXT
);

-- ══════════════════════════════════════════════════════════════════
--  CLEAN MARITIME IMPORTS  (~15.5M rows)
--  Maritime only: cod_via_transporte = 1
--  Already decoded: codes replaced with human-readable labels
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS clean_maritime_imports (
    id                      BIGSERIAL PRIMARY KEY,
    periodo                 SMALLINT,          -- year  (nullable: 754K rows have NULL)
    mes                     SMALLINT,          -- month (nullable: same rows)
    aduana                  TEXT,
    regimen_importacion     TEXT,
    sigla_regimen           TEXT,
    pais_origen             TEXT,
    continente_origen       TEXT,
    puerto_embarque         TEXT,
    zona_geo_embarque       TEXT,
    puerto_desembarque      TEXT,
    tipo_carga              TEXT,
    clausula_compra         TEXT,
    sigla_clausula          TEXT,
    item_sa                 TEXT,
    hs6_subpartida          TEXT,
    hs4_partida             TEXT,
    hs2_capitulo            TEXT,
    descripcion_producto    TEXT,
    cif_us                  DOUBLE PRECISION,
    cantidad_mercancia      DOUBLE PRECISION,
    unidad_medida           TEXT,
    sigla_unidad            TEXT
);

-- Indexes covering 95% of analytical query patterns
CREATE INDEX IF NOT EXISTS idx_cmi_periodo_mes    ON clean_maritime_imports (periodo, mes);
CREATE INDEX IF NOT EXISTS idx_cmi_port_dest      ON clean_maritime_imports (puerto_desembarque);
CREATE INDEX IF NOT EXISTS idx_cmi_hs2            ON clean_maritime_imports (hs2_capitulo);
CREATE INDEX IF NOT EXISTS idx_cmi_hs6            ON clean_maritime_imports (hs6_subpartida);
CREATE INDEX IF NOT EXISTS idx_cmi_pais           ON clean_maritime_imports (pais_origen);
CREATE INDEX IF NOT EXISTS idx_cmi_port_period    ON clean_maritime_imports (puerto_desembarque, periodo, mes);
CREATE INDEX IF NOT EXISTS idx_cmi_hs2_period     ON clean_maritime_imports (hs2_capitulo, periodo);

-- ══════════════════════════════════════════════════════════════════
--  CLEAN MARITIME EXPORTS  (~4.5M rows)
--  Maritime only: cod_via_transporte = 1
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS clean_maritime_exports (
    id                      BIGSERIAL PRIMARY KEY,
    periodo                 SMALLINT,          -- nullable (same reason as imports)
    mes                     SMALLINT,
    aduana                  TEXT,
    region_origen           TEXT,
    puerto_embarque         TEXT,
    zona_geo_embarque       TEXT,
    pais_destino            TEXT,
    continente_destino      TEXT,
    puerto_desembarque      TEXT,
    tipo_carga              TEXT,
    clausula_venta          TEXT,
    sigla_clausula          TEXT,
    item_sa                 TEXT,
    hs6_subpartida          TEXT,
    hs4_partida             TEXT,
    hs2_capitulo            TEXT,
    descripcion_producto    TEXT,
    fob_us                  DOUBLE PRECISION,
    peso_bruto_kg           DOUBLE PRECISION,
    cantidad_mercancia      DOUBLE PRECISION,
    unidad_medida           TEXT,
    sigla_unidad            TEXT
);

CREATE INDEX IF NOT EXISTS idx_cme_periodo_mes    ON clean_maritime_exports (periodo, mes);
CREATE INDEX IF NOT EXISTS idx_cme_port_emb       ON clean_maritime_exports (puerto_embarque);
CREATE INDEX IF NOT EXISTS idx_cme_hs2            ON clean_maritime_exports (hs2_capitulo);
CREATE INDEX IF NOT EXISTS idx_cme_hs6            ON clean_maritime_exports (hs6_subpartida);
CREATE INDEX IF NOT EXISTS idx_cme_pais           ON clean_maritime_exports (pais_destino);
CREATE INDEX IF NOT EXISTS idx_cme_port_period    ON clean_maritime_exports (puerto_embarque, periodo, mes);
CREATE INDEX IF NOT EXISTS idx_cme_hs2_period     ON clean_maritime_exports (hs2_capitulo, periodo);

-- ══════════════════════════════════════════════════════════════════
--  ML — PORT CONGESTION TABLES  (5 tables)
--  Built by 03_ml_congestion.sql from the clean tables above
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS port_monthly_agg (
    year                        INTEGER,
    month                       INTEGER,
    port                        TEXT,
    direction                   TEXT,
    aduana                      TEXT,
    shipment_count              BIGINT,
    total_value_usd             DOUBLE PRECISION,
    total_weight_mt             DOUBLE PRECISION,
    total_quantity              DOUBLE PRECISION,
    commodity_diversity         BIGINT,
    hs4_diversity               BIGINT,
    country_diversity           BIGINT,
    continent_diversity         BIGINT,
    dominant_hs2                TEXT,
    dominant_cargo_type         TEXT,
    dominant_origin_country     TEXT,
    dominant_continent          TEXT,
    cnt_general                 BIGINT,       -- DuckDB HUGEINT → PostgreSQL BIGINT
    cnt_bulk                    BIGINT,
    cnt_refrigerated            BIGINT,
    cnt_container               BIGINT,
    avg_value_per_shipment_usd  DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_pma_port_dir_ym ON port_monthly_agg (port, direction, year, month);

-- ── port_features_indexed — 49 columns ───────────────────────────
CREATE TABLE IF NOT EXISTS port_features_indexed (
    year                        INTEGER,
    month                       INTEGER,
    port                        TEXT,
    direction                   TEXT,
    aduana                      TEXT,
    shipment_count              BIGINT,
    total_value_usd             DOUBLE PRECISION,
    total_weight_mt             DOUBLE PRECISION,
    total_quantity              DOUBLE PRECISION,
    commodity_diversity         BIGINT,
    hs4_diversity               BIGINT,
    country_diversity           BIGINT,
    continent_diversity         BIGINT,
    dominant_hs2                TEXT,
    dominant_cargo_type         TEXT,
    dominant_origin_country     TEXT,
    dominant_continent          TEXT,
    cnt_general                 BIGINT,
    cnt_bulk                    BIGINT,
    cnt_refrigerated            BIGINT,
    cnt_container               BIGINT,
    avg_value_per_shipment_usd  DOUBLE PRECISION,
    quarter                     DOUBLE PRECISION,
    season                      TEXT,
    year_index                  INTEGER,
    month_sin                   DOUBLE PRECISION,
    month_cos                   DOUBLE PRECISION,
    pct_general                 DOUBLE PRECISION,
    pct_bulk                    DOUBLE PRECISION,
    pct_refrigerated            DOUBLE PRECISION,
    pct_container               DOUBLE PRECISION,
    weight_per_shipment_mt      DOUBLE PRECISION,
    avg_quantity_per_shipment   DOUBLE PRECISION,
    lag_1                       BIGINT,
    lag_2                       BIGINT,
    lag_3                       BIGINT,
    lag_12                      BIGINT,
    lag_value_12                DOUBLE PRECISION,
    lag_weight_12               DOUBLE PRECISION,
    rolling_3_mean              DOUBLE PRECISION,
    rolling_12_mean             DOUBLE PRECISION,
    rolling_value_12_mean       DOUBLE PRECISION,
    yoy_growth                  DOUBLE PRECISION,
    yoy_value_growth            DOUBLE PRECISION,
    sc_norm                     DOUBLE PRECISION,
    v_norm                      DOUBLE PRECISION,
    w_norm                      DOUBLE PRECISION,
    cd_norm                     DOUBLE PRECISION,
    congestion_index            DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_pfi_port_dir_ym ON port_features_indexed (port, direction, year, month);

CREATE TABLE IF NOT EXISTS port_seasonal_index (
    port                    TEXT,
    direction               TEXT,
    month                   INTEGER,
    seasonal_factor         DOUBLE PRECISION,
    seasonal_weight_factor  DOUBLE PRECISION,
    hist_avg_congestion     DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_psi_port_dir_m ON port_seasonal_index (port, direction, month);

-- ── port_forecast_params — 22 columns ────────────────────────────
CREATE TABLE IF NOT EXISTS port_forecast_params (
    port                        TEXT,
    direction                   TEXT,
    last_year                   INTEGER,
    last_month                  INTEGER,
    avg_yoy_growth              DOUBLE PRECISION,
    avg_yoy_value_growth        DOUBLE PRECISION,
    avg_shipments               DOUBLE PRECISION,
    avg_value_usd               DOUBLE PRECISION,
    avg_weight_mt               DOUBLE PRECISION,
    avg_commodity_diversity     DOUBLE PRECISION,
    avg_hs4_diversity           DOUBLE PRECISION,
    avg_country_diversity       DOUBLE PRECISION,
    avg_pct_general             DOUBLE PRECISION,
    avg_pct_bulk                DOUBLE PRECISION,
    avg_pct_refrigerated        DOUBLE PRECISION,
    avg_pct_container           DOUBLE PRECISION,
    avg_weight_per_shipment_mt  DOUBLE PRECISION,
    avg_qty_per_shipment        DOUBLE PRECISION,
    dominant_hs2                TEXT,
    dominant_cargo_type         TEXT,
    dominant_continent          TEXT,
    n_obs                       BIGINT
);

-- ── port_congestion_forecast — 23 columns ────────────────────────
CREATE TABLE IF NOT EXISTS port_congestion_forecast (
    port                    TEXT,
    direction               TEXT,
    year                    INTEGER,
    month                   INTEGER,
    season                  TEXT,
    pred_shipment_count     INTEGER,
    pred_weight_mt          DOUBLE PRECISION,
    pred_total_quantity     DOUBLE PRECISION,
    pred_value_usd          DOUBLE PRECISION,
    congestion_index        DOUBLE PRECISION,
    congestion_label        TEXT,
    avg_commodity_diversity DOUBLE PRECISION,
    avg_hs4_diversity       DOUBLE PRECISION,
    avg_country_diversity   DOUBLE PRECISION,
    pct_general             DOUBLE PRECISION,
    pct_bulk                DOUBLE PRECISION,
    pct_refrigerated        DOUBLE PRECISION,
    pct_container           DOUBLE PRECISION,
    dominant_commodity_hs2  TEXT,
    dominant_cargo_type     TEXT,
    dominant_continent      TEXT,
    model_version           TEXT,
    forecast_generated_at   TIMESTAMPTZ
);

-- ══════════════════════════════════════════════════════════════════
--  ML — COMMODITY TABLES  (3 tables)
--  Built by 04_ml_commodity.sql
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS commodity_port_params (
    hs2                     TEXT,
    port                    TEXT,
    n_obs                   BIGINT,
    last_year               INTEGER,
    last_month              INTEGER,
    avg_monthly_fob         DOUBLE PRECISION,
    avg_monthly_weight_mt   DOUBLE PRECISION,
    avg_monthly_ships       DOUBLE PRECISION,
    avg_yoy_fob_growth      DOUBLE PRECISION,
    avg_yoy_weight_growth   DOUBLE PRECISION,
    avg_yoy_ships_growth    DOUBLE PRECISION,
    dominant_description    TEXT
);

CREATE INDEX IF NOT EXISTS idx_cpp_hs2_port ON commodity_port_params (hs2, port);

CREATE TABLE IF NOT EXISTS commodity_seasonal_index (
    hs2                     TEXT,
    port                    TEXT,
    month                   INTEGER,
    avg_monthly_fob         DOUBLE PRECISION,
    seasonal_fob_factor     DOUBLE PRECISION,
    seasonal_weight_factor  DOUBLE PRECISION,
    overall_avg_ships       DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_csi_hs2_port_m ON commodity_seasonal_index (hs2, port, month);

CREATE TABLE IF NOT EXISTS commodity_port_forecast (
    hs2_capitulo            TEXT,
    puerto_embarque         TEXT,
    year                    INTEGER,
    month                   INTEGER,
    season                  TEXT,
    pred_fob_usd            DOUBLE PRECISION,
    pred_weight_mt          DOUBLE PRECISION,
    pred_shipment_count     INTEGER,
    season_activity_index   DOUBLE PRECISION,
    season_label            TEXT,
    commodity_description   TEXT,
    historical_months       BIGINT,
    model_version           TEXT,
    forecast_generated_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_cpf_hs2_port_ym
    ON commodity_port_forecast (hs2_capitulo, puerto_embarque, year, month);

-- ══════════════════════════════════════════════════════════════════
--  ML PIPELINE OUTPUT TABLES  (4 tables)
--  Written by 05_ml_train_evaluate.py after training
-- ══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS ml_model_evaluation (
    model           TEXT,
    port            TEXT,
    direction       TEXT,
    fold_year       INTEGER,
    n_samples       INTEGER,
    mae             DOUBLE PRECISION,
    rmse            DOUBLE PRECISION,
    mape            DOUBLE PRECISION,
    r2              DOUBLE PRECISION,
    evaluated_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ml_feature_importance (
    port            TEXT,
    direction       TEXT,
    feature         TEXT,
    shap_mean_abs   DOUBLE PRECISION,
    lgbm_gain       DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS ml_forecast_2026 (
    port                    TEXT,
    direction               TEXT,
    year                    INTEGER,
    month                   INTEGER,
    pred_shipment_count     INTEGER,
    best_model              TEXT,
    forecast_generated_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ml_covid_diagnostics (
    port                        TEXT,
    direction                   TEXT,
    pre_covid_avg_ships         DOUBLE PRECISION,
    covid_2020_avg_ships        DOUBLE PRECISION,
    covid_2021_avg_ships        DOUBLE PRECISION,
    post_covid_2023_ships       DOUBLE PRECISION,
    pct_drop_2020               DOUBLE PRECISION,
    pct_rebound_2021            DOUBLE PRECISION,
    n_contaminated_lag12_rows   INTEGER
);

-- ── Summary ───────────────────────────────────────────────────────
SELECT
    schemaname,
    tablename,
    'created' AS status
FROM pg_tables
WHERE schemaname = 'waze_cargo'
ORDER BY tablename;
