# WazeCargo — Database Schema Reference

**Database:** AWS RDS PostgreSQL 17.6  
**Host:** `wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com`  
**Total size:** ~15.2 GB across 35 tables in 5 schemas

---

## Data Pipeline Flow

```
structured.all_imports (33.8M)  ──┐
structured.all_exports (7.1M)   ──┤
13 lookup tables                ──┤
                                  ▼
                    ┌─────────────────────────────┐
                    │  03_rebuild_clean_maritime   │
                    │  (filter maritime + JOINs)   │
                    └──────────────┬───────────────┘
                                  ▼
              maritime.clean_maritime_imports (14.6M)
              maritime.clean_maritime_exports (4.5M)
                    │                       │
          ┌────────┘                       └────────┐
          ▼                                         ▼
┌───────────────────┐                   ┌────────────────────┐
│ 04_ml_congestion  │                   │  05_ml_commodity   │
│ (port-level ML)   │                   │ (hs2+port-level ML)│
└────────┬──────────┘                   └─────────┬──────────┘
         ▼                                        ▼
  ml.port_monthly_agg                    ml.commodity_monthly_agg
  ml.port_features_indexed               ml.commodity_features
  ml.port_cv_metrics                     ml.commodity_cv_metrics
  ml.port_model_selection                ml.commodity_model_selection
  ml.port_forecast_2026                  ml.commodity_forecast_2026

                    Weather pipeline (external)
                    ─────────────────────────────
                    waze_cargo.port_risk_config
                         ▼
                    waze_cargo.port_weather_hourly (9.2M)
                         ▼
                    waze_cargo.port_weather_daily (383K)
                         ▼
                    waze_cargo.port_weather_monthly (12.6K)
                         ▼
                    waze_cargo.port_operability_score (2.7M)
                         ▼
                    waze_cargo.port_congestion_weather_adjusted (12.7K)
                         ▲
                    ml.port_features_indexed ────┘
```

---

## Schema: `structured` (15 tables, ~5.2 GB)

Raw Chilean customs data + lookup tables. Loaded by `pipeline/02_staging_to_structured.py`.

### structured.all_imports (~33.8M rows, 3.8 GB)

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| PERIODO | integer | YES | Year |
| MES | integer | YES | Month |
| COD_ADUANA_TRAMITACION | text | YES | FK → lkp_aduanas |
| COD_TIPO_OPERACION | text | YES | FK → lkp_tipos_operacion |
| COD_PAIS_ORIGEN | text | YES | FK → lkp_paises |
| COD_PAIS_ADQUISICION | text | YES | |
| COD_REGIMEN_IMPORTACION | text | YES | FK → lkp_regimen_importacion |
| COD_PUERTO_EMBARQUE | text | YES | FK → lkp_puertos |
| COD_PUERTO_DESEMBARQUE | text | YES | FK → lkp_puertos |
| COD_VIA_TRANSPORTE | text | YES | FK → lkp_vias_transporte |
| CL_COMPRA | text | YES | FK → lkp_clausulas |
| ITEM_SA | text | YES | FK → lkp_harmonized_system (via HS6) |
| CIF_US | text | YES | Cost/Insurance/Freight USD (stored as text) |
| AD_VALOREM_US | text | YES | Ad valorem tax USD |
| MONEDA | text | YES | FK → lkp_moneda |
| CANTIDAD_MERCANCIA | text | YES | Quantity |
| COD_UNIDAD_MEDIDA | text | YES | FK → lkp_unidades_medida |
| TPO_CARGA | text | YES | FK → lkp_tipos_carga |
| year | integer | YES | Partition year |

**Indexes:** `idx_all_imports_year (year)`

### structured.all_exports (~7.1M rows, 1.4 GB)

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| PERIODO | integer | YES | Year |
| MES | integer | YES | Month |
| COD_ADUANA_TRAMITACION | text | YES | FK → lkp_aduanas |
| COD_TIPO_OPERACION | text | YES | FK → lkp_tipos_operacion |
| COD_REGION_ORIGEN | text | YES | FK → lkp_regiones |
| COD_VIA_TRANSPORTE | text | YES | FK → lkp_vias_transporte |
| COD_PUERTO_EMBARQUE | text | YES | FK → lkp_puertos |
| COD_PUERTO_DESEMBARQUE | text | YES | FK → lkp_puertos |
| COD_PAIS_DESTINO | text | YES | FK → lkp_paises |
| COD_MODALIDAD_VENTA | text | YES | FK → lkp_modalidades_venta |
| MONEDA | text | YES | FK → lkp_moneda |
| CLAUSULA_VENTA | text | YES | FK → lkp_clausulas |
| COD_TIPO_CARGA | text | YES | FK → lkp_tipos_carga |
| ITEM_SA | text | YES | FK → lkp_harmonized_system (via HS6) |
| FOB_US_DUSLEG | text | YES | FOB value USD |
| FOBUS_AJUSTADO_IVV | text | YES | Adjusted FOB |
| PESO_BRUTO_KG | text | YES | Gross weight kg |
| CANTIDAD_MERCANCIA | text | YES | Quantity |
| COD_UNIDAD_MEDIDA | text | YES | FK → lkp_unidades_medida |
| year | integer | YES | Partition year |

**Indexes:** `idx_all_exports_year (year)`

### Lookup Tables (13 tables)

| Table | PK Column | PK Type | Key Columns |
|-------|-----------|---------|-------------|
| lkp_aduanas | cod_aduana_tramitacion | integer | nombre_aduana |
| lkp_clausulas | cl_compra | integer | nombre_clausula, sigla_clausula |
| lkp_harmonized_system | hscode | text | section, description, parent, level |
| lkp_modalidades_venta | cod_modalidad_venta | integer | nombre_modalidad_venta |
| lkp_moneda | moneda | integer | moneda_1, pais_moneda |
| lkp_paises | cod_pais | integer | nombre_pais, nombre_continente |
| lkp_puertos | cod_puerto | integer | nombre_puerto, tipo_puerto, zona_geografica |
| lkp_regimen_importacion | cod_rgimen_importacion | integer | nombre_rgimen_importacion, sigla |
| lkp_regiones | cod_region_origen | integer | nombre_region |
| lkp_tipos_carga | cod_tipo_carga | text | nombre_tipo_carga |
| lkp_tipos_operacion | cod_tipo_operacion | integer | nombre_tipo_operacion, ingreso_salida |
| lkp_unidades_medida | cod_unidad_medida | integer | unidad_medida, nombre_unidad_medida |
| lkp_vias_transporte | cod_via_transporte | integer | nombre_via_transporte |

---

## Schema: `maritime` (2 tables, ~6.3 GB)

Clean, decoded, maritime-only records. Built by `modeling/03_rebuild_clean_maritime.sh` from structured + lookups. Filters: `cod_via_transporte = 1` and excludes airports.

### maritime.clean_maritime_imports (~14.6M rows, 4.9 GB)

| Column | Type | Nullable | Source |
|--------|------|----------|--------|
| periodo | smallint | YES | all_imports.PERIODO |
| mes | smallint | YES | all_imports.MES |
| aduana | text | YES | lkp_aduanas.nombre_aduana |
| regimen_importacion | text | YES | lkp_regimen_importacion.nombre |
| sigla_regimen | text | YES | lkp_regimen_importacion.sigla |
| pais_origen | text | YES | lkp_paises.nombre_pais |
| continente_origen | text | YES | lkp_paises.nombre_continente |
| puerto_embarque | text | YES | lkp_puertos.nombre_puerto (origin) |
| zona_geo_embarque | text | YES | lkp_puertos.zona_geografica |
| puerto_desembarque | text | YES | lkp_puertos.nombre_puerto (destination) |
| tipo_carga | text | YES | lkp_tipos_carga.nombre_tipo_carga |
| clausula_compra | text | YES | lkp_clausulas.nombre_clausula |
| sigla_clausula | text | YES | lkp_clausulas.sigla_clausula |
| item_sa | text | YES | Raw HS code |
| hs6_subpartida | text | YES | LEFT(item_sa, 6) |
| hs4_partida | text | YES | LEFT(item_sa, 4) |
| hs2_capitulo | text | YES | LEFT(item_sa, 2) |
| descripcion_producto | text | YES | lkp_harmonized_system.description |
| cif_us | double precision | YES | Cast from text |
| cantidad_mercancia | double precision | YES | Cast from text |
| unidad_medida | text | YES | lkp_unidades_medida.nombre |
| sigla_unidad | text | YES | lkp_unidades_medida.unidad_medida |

### maritime.clean_maritime_exports (~4.5M rows, 1.4 GB)

| Column | Type | Nullable | Source |
|--------|------|----------|--------|
| periodo | smallint | YES | all_exports.PERIODO |
| mes | smallint | YES | all_exports.MES |
| aduana | text | YES | lkp_aduanas.nombre_aduana |
| region_origen | text | YES | lkp_regiones.nombre_region |
| puerto_embarque | text | YES | lkp_puertos.nombre_puerto (origin) |
| zona_geo_embarque | text | YES | lkp_puertos.zona_geografica |
| pais_destino | text | YES | lkp_paises.nombre_pais |
| continente_destino | text | YES | lkp_paises.nombre_continente |
| puerto_desembarque | text | YES | lkp_puertos.nombre_puerto (destination) |
| tipo_carga | text | YES | lkp_tipos_carga.nombre_tipo_carga |
| clausula_venta | text | YES | lkp_clausulas.nombre_clausula |
| sigla_clausula | text | YES | lkp_clausulas.sigla_clausula |
| item_sa | text | YES | Raw HS code |
| hs6_subpartida | text | YES | LEFT(item_sa, 6) |
| hs4_partida | text | YES | LEFT(item_sa, 4) |
| hs2_capitulo | text | YES | LEFT(item_sa, 2) |
| descripcion_producto | text | YES | lkp_harmonized_system.description |
| fob_us | double precision | YES | Cast from text |
| peso_bruto_kg | double precision | YES | Cast from text |
| cantidad_mercancia | double precision | YES | Cast from text |
| unidad_medida | text | YES | lkp_unidades_medida.nombre |
| sigla_unidad | text | YES | lkp_unidades_medida.unidad_medida |

---

## Schema: `ml` (10 tables, ~163 MB)

ML model outputs. Built by `modeling/04_ml_congestion.py` (port-level) and `modeling/05_ml_commodity.py` (commodity-level).

### Port Congestion Pipeline (5 tables)

#### ml.port_monthly_agg (~13.7K rows)

Aggregated from maritime.clean_maritime_imports/exports by (port, direction, year, month).

| Column | Type | Notes |
|--------|------|-------|
| year | integer | |
| month | integer | |
| port | text | Puerto name |
| direction | text | "import" or "export" |
| shipment_count | bigint | Count of shipments |
| total_value_usd | double precision | Sum CIF/FOB |
| total_weight_mt | double precision | Sum weight in metric tons |
| total_quantity | double precision | Sum quantity |
| commodity_diversity | bigint | Count distinct HS2 |
| hs4_diversity | bigint | Count distinct HS4 |
| country_diversity | bigint | Count distinct countries |
| continent_diversity | bigint | Count distinct continents |
| dominant_hs2 | text | Most frequent HS2 chapter |
| dominant_cargo_type | text | Most frequent cargo type |
| dominant_origin_country | text | Most frequent origin/dest |
| dominant_continent | text | Most frequent continent |
| cnt_general | bigint | General cargo count |
| cnt_bulk | bigint | Bulk cargo count |
| cnt_refrigerated | bigint | Refrigerated cargo count |
| cnt_container | bigint | Container cargo count |
| avg_value_per_shipment_usd | double precision | |

**Index:** `(port, direction, year, month)`

#### ml.port_features_indexed (~12.7K rows)

Feature-engineered table with lags, rolling averages, COVID flags, and congestion index.

| Column | Type | Notes |
|--------|------|-------|
| year, month, port, direction | — | Same grain as port_monthly_agg |
| All port_monthly_agg columns | — | Carried forward |
| season | text | Southern hemisphere season |
| quarter | double precision | |
| year_index | integer | Sequential year number |
| month_sin, month_cos | double precision | Cyclical month encoding |
| pct_general/bulk/refrigerated/container | numeric | Cargo type percentages |
| weight_per_shipment_mt | double precision | |
| lag_1, lag_2, lag_3, lag_12, _lag_24, _lag_36, _lag_48 | bigint | Shipment count lags |
| lag_value_12, lag_weight_12 | double precision | Value/weight 12-month lags |
| rolling_3_mean, rolling_12_mean | double precision | Rolling averages |
| rolling_value_12_mean | double precision | |
| yoy_growth, yoy_value_growth | double precision | Year-over-year growth |
| is_covid_shock | integer | Flag: 2020 COVID drop |
| is_covid_rebound | integer | Flag: 2021 rebound |
| is_covid_aftershock | integer | Flag: 2022 aftershock |
| lag_12_is_covid | integer | Whether lag_12 was COVID |
| lag_12_clean, yoy_growth_clean | — | COVID-cleaned versions |
| sc_norm, v_norm, w_norm, cd_norm | — | Normalized components |
| congestion_index | double precision | Composite congestion score |

**Index:** `(port, direction, year, month)`

#### ml.port_cv_metrics (~1.6K rows)

Cross-validation results per (port, direction, model, fold_year).

| Column | Type |
|--------|------|
| port, direction, model | text |
| fold_year | bigint |
| mae, rmse, mape, r2 | double precision |
| n, features_used | bigint |
| created_at | timestamptz |

#### ml.port_model_selection (~61 rows)

Best model per (port, direction).

| Column | Type |
|--------|------|
| port, direction | text |
| avg_volume | double precision |
| selected_model | text |
| best_ml_model | text |
| ml_cv_mape, baseline_cv_mape, final_cv_mape | double precision |
| volume_threshold | bigint |
| created_at | timestamptz |

#### ml.port_forecast_2026 (~666 rows)

12-month forecast per (port, direction).

| Column | Type |
|--------|------|
| year, month | bigint |
| port, direction | text |
| pred_shipment_count | double precision |
| model | text |
| forecast_shipments | bigint |
| created_at | timestamptz |

### Commodity Pipeline (5 tables)

#### ml.commodity_monthly_agg (~231K rows)

Aggregated by (hs2, port, direction, year, month).

| Column | Type |
|--------|------|
| hs2, port, direction | text |
| year, month | integer |
| shipment_count | bigint |
| total_value_usd, total_weight_mt, total_quantity | double precision |
| dominant_description | text |

**Index:** `(hs2, port, direction, year, month)`

#### ml.commodity_features (~208K rows)

Feature-engineered with same lag/rolling/COVID pattern as port_features_indexed, plus hs2 dimension.

**Index:** `(hs2, port, direction, year, month)`

#### ml.commodity_cv_metrics (~30.6K rows)

Cross-validation per (hs2, port, direction, model, fold_year).

#### ml.commodity_model_selection (~1.2K rows)

Best model per (hs2, port, direction).

#### ml.commodity_forecast_2026 (~12.8K rows)

12-month forecast per (hs2, port, direction).

---

## Schema: `waze_cargo` (6 tables, ~3.5 GB)

Weather data and port operability scoring.

### waze_cargo.port_risk_config

Per-port thresholds for weather risk scoring.

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| port_code | varchar | **PK** | e.g. "CLVAP" |
| zone | varchar | NO | Coastal zone |
| hs_threshold_m | real | NO | Wave warning threshold |
| hs_closure_m | real | NO | Wave closure threshold |
| swell_sensitivity | real | NO | |
| exposure_factor | real | NO | Port exposure to open sea |
| wind_threshold_kmh | real | NO | Wind warning threshold |
| wind_closure_kmh | real | NO | Wind closure threshold |
| gusts_threshold_kmh | real | NO | Gust warning threshold |
| gusts_closure_kmh | real | NO | Gust closure threshold |
| notes | text | YES | |

### waze_cargo.port_weather_hourly (~9.2M rows, 2.1 GB)

Hourly marine weather observations/forecasts per port.

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| port_code | varchar | **PK** | |
| ts | timestamp | **PK** | Hour timestamp |
| cod_puerto | integer | NO | FK → lkp_puertos |
| port_name | varchar | YES | |
| zone | varchar | YES | |
| year, month, day, hour | int/smallint | NO | Extracted from ts |
| wave_height_m | real | YES | Significant wave height |
| wave_period_s | real | YES | |
| wave_direction_deg | smallint | YES | |
| swell_height_m | real | YES | |
| swell_period_s | real | YES | |
| swell_direction_deg | smallint | YES | |
| wind_speed_kmh | real | YES | |
| wind_gusts_kmh | real | YES | |
| wind_direction_deg | smallint | YES | |
| wave_risk_score | real | YES | Computed risk |
| wind_risk_score | real | YES | Computed risk |
| combined_risk | real | YES | |
| operability_index | real | YES | 0-100 score |
| risk_label | varchar | YES | "normal"/"watch"/"warning"/"closed" |
| closure_driver | varchar | YES | What caused closure |
| swell_dominant | boolean | YES | |

**Indexes:** `PK(port_code, ts)`, `(cod_puerto, ts)`, `(risk_label)`, `(year, month)`

### waze_cargo.port_weather_daily (~383K rows, 82 MB)

Daily rollup from hourly data.

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| port_code | varchar | **PK** | |
| date | date | **PK** | |
| cod_puerto | integer | NO | FK → lkp_puertos |
| port_name, zone | varchar | YES | |
| year, month, day | int/smallint | NO | |
| n_hours | smallint | YES | Hours with data |
| n_closed_hours | smallint | YES | |
| n_warning_hours | smallint | YES | |
| n_advisory_hours | smallint | YES | |
| n_watch_hours | smallint | YES | |
| n_normal_hours | smallint | YES | |
| n_closed_by_wave/swell/wind/gust | smallint | YES | Closure driver counts |
| pct_hours_closed | real | YES | % of day closed |
| pct_hours_warning | real | YES | |
| avg_operability_index | real | YES | Daily average |
| min_operability_index | real | YES | Worst hour |
| max_wave_risk_score | real | YES | |
| max_wind_risk_score | real | YES | |
| max_wave_height_m | real | YES | |
| avg_wave_height_m | real | YES | |
| max_swell_height_m | real | YES | |
| max_wind_speed_kmh | real | YES | |
| max_wind_gusts_kmh | real | YES | |
| is_closure_day | boolean | YES | Any closed hour |

**Indexes:** `PK(port_code, date)`, `(cod_puerto, date)`, `(year, month)`

### waze_cargo.port_weather_monthly (~12.6K rows, 3.2 MB)

Monthly rollup from daily data.

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| port_code | varchar | **PK** | |
| year | integer | **PK** | |
| month | smallint | **PK** | |
| cod_puerto | integer | NO | FK → lkp_puertos |
| port_name, zone | varchar | YES | |
| n_hours, n_closed_hours, n_warning_hours | integer | YES | |
| n_closed_by_wave/swell/wind/gust | integer | YES | |
| pct_hours_closed | real | YES | |
| pct_hours_warning | real | YES | |
| n_closure_days | smallint | YES | |
| avg_operability_index | real | YES | |
| avg_wave_risk_score | real | YES | |
| avg_wind_risk_score | real | YES | |
| max/avg wave/swell/wind heights | real | YES | |

**Indexes:** `PK(port_code, year, month)`, `(cod_puerto, year, month)`, `(year, month)`

### waze_cargo.port_operability_score (~2.7M rows, 1.3 GB)

Scored operability per port per hour (includes forecasts).

| Column | Type | Nullable | Notes |
|--------|------|----------|-------|
| id | bigserial | **PK** | Auto-increment |
| port | text | NO | Port name |
| zone | text | NO | |
| ts | timestamptz | NO | UNIQUE with port |
| wave_height_m | real | YES | |
| wave_direction_deg | real | YES | |
| wave_period_s | real | YES | |
| swell_height_m | real | YES | |
| swell_direction_deg | real | YES | |
| swell_period_s | real | YES | |
| wind_wave_height_m | real | YES | |
| wind_speed_kmh | real | YES | |
| wind_direction_deg | real | YES | |
| wind_gust_kmh | real | YES | |
| exposure_factor | real | YES | From port_risk_config |
| wave_risk_score | real | YES | |
| wind_risk_score | real | YES | |
| operability_index | real | NO | 0-100 |
| risk_level | smallint | NO | 0-4 |
| risk_label | text | NO | normal/watch/advisory/warning/closed |
| closure_driver | text | YES | |
| swell_dominant | boolean | YES | |
| hs_threshold_m | real | YES | From config |
| hs_closure_m | real | YES | From config |
| is_forecast | boolean | NO | Default false |
| scored_at | timestamptz | NO | Default now() |

**Indexes:** `PK(id)`, `UNIQUE(port, ts)`, `(port, risk_level, ts)`

### waze_cargo.port_congestion_weather_adjusted (~12.7K rows, 8.3 MB)

Port congestion index adjusted by weather operability. Joins ml.port_features_indexed with waze_cargo.port_weather_monthly.

| Column | Type | Notes |
|--------|------|-------|
| year, month | integer | |
| port, direction | text | From port_features_indexed |
| All port_features_indexed columns | — | Carried forward |
| congestion_index_raw | double precision | Before weather adjustment |
| port_code | text | Matched port code |
| pct_hours_closed | real | From port_weather_monthly |
| pct_hours_warning | real | From port_weather_monthly |
| adjustment_multiplier | real | Weather penalty factor |
| adjustment_type | text | "none"/"minor"/"moderate"/"severe" |
| computed_at | timestamptz | |

**Indexes:** `(port_code, year, month, direction)`, `(port, year, month)`, `(adjustment_type)`, `(year, month)`

---

## Schema: `wazecargo_dev` (2 tables, empty)

Development copies of maritime tables. Same schema as `maritime.clean_maritime_imports` and `maritime.clean_maritime_exports`. Currently empty.

---

## Implicit Relationships (no enforced FKs)

The database uses **logical foreign keys** (no `FOREIGN KEY` constraints enforced). Relationships are maintained by the pipeline scripts:

| Source Table | Column | Target Table | Target Column |
|---|---|---|---|
| structured.all_imports | COD_ADUANA_TRAMITACION | structured.lkp_aduanas | cod_aduana_tramitacion |
| structured.all_imports | COD_PAIS_ORIGEN | structured.lkp_paises | cod_pais |
| structured.all_imports | COD_PUERTO_EMBARQUE | structured.lkp_puertos | cod_puerto |
| structured.all_imports | COD_PUERTO_DESEMBARQUE | structured.lkp_puertos | cod_puerto |
| structured.all_imports | COD_REGIMEN_IMPORTACION | structured.lkp_regimen_importacion | cod_rgimen_importacion |
| structured.all_imports | CL_COMPRA | structured.lkp_clausulas | cl_compra |
| structured.all_imports | TPO_CARGA | structured.lkp_tipos_carga | cod_tipo_carga |
| structured.all_imports | COD_VIA_TRANSPORTE | structured.lkp_vias_transporte | cod_via_transporte |
| structured.all_imports | COD_UNIDAD_MEDIDA | structured.lkp_unidades_medida | cod_unidad_medida |
| structured.all_imports | ITEM_SA (HS6) | structured.lkp_harmonized_system | hscode |
| structured.all_exports | COD_ADUANA_TRAMITACION | structured.lkp_aduanas | cod_aduana_tramitacion |
| structured.all_exports | COD_PAIS_DESTINO | structured.lkp_paises | cod_pais |
| structured.all_exports | COD_PUERTO_EMBARQUE | structured.lkp_puertos | cod_puerto |
| structured.all_exports | COD_PUERTO_DESEMBARQUE | structured.lkp_puertos | cod_puerto |
| structured.all_exports | COD_REGION_ORIGEN | structured.lkp_regiones | cod_region_origen |
| structured.all_exports | CLAUSULA_VENTA | structured.lkp_clausulas | cl_compra |
| structured.all_exports | COD_TIPO_CARGA | structured.lkp_tipos_carga | cod_tipo_carga |
| structured.all_exports | COD_VIA_TRANSPORTE | structured.lkp_vias_transporte | cod_via_transporte |
| structured.all_exports | COD_UNIDAD_MEDIDA | structured.lkp_unidades_medida | cod_unidad_medida |
| structured.all_exports | COD_MODALIDAD_VENTA | structured.lkp_modalidades_venta | cod_modalidad_venta |
| structured.all_exports | ITEM_SA (HS6) | structured.lkp_harmonized_system | hscode |
| waze_cargo.port_weather_* | cod_puerto | structured.lkp_puertos | cod_puerto |
| waze_cargo.port_congestion_weather_adjusted | (port, year, month) | ml.port_features_indexed | (port, year, month) |
| waze_cargo.port_congestion_weather_adjusted | (port_code, year, month) | waze_cargo.port_weather_monthly | (port_code, year, month) |
