"""
WAZE CARGO — Feature Cache Builder
==================================

Reads waze_cargo.duckdb (raw clean_maritime_imports / clean_maritime_exports)
and materialises the SAME feature tables that the production pipeline writes
to PostgreSQL:

    port_monthly_agg          (one row per port × direction × year × month)
    port_features_indexed     (lag, rolling, COVID flags, congestion index)

Output: parquet files in ./data/  consumed by every EDA notebook.

Run once:
    python build_feature_cache.py
"""

import os
import time
import duckdb
import numpy as np
import pandas as pd

DB_PATH   = "/home/koko/Waze_Cargo/waze_cargo.duckdb"
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

COVID_YEARS      = {2020, 2021, 2022}
INCOMPLETE_YEARS = {2024}
CLEAN_YEARS      = set(range(2005, 2026)) - COVID_YEARS - INCOMPLETE_YEARS

print(f"Connecting to DuckDB: {DB_PATH}")
con = duckdb.connect(DB_PATH, read_only=True)

# ── 1. PORT MONTHLY AGGREGATE ─────────────────────────────────────
t0 = time.time()
print("Step 1/2 — Building port_monthly_agg ...")

agg_sql = """
SELECT
    i.periodo                                        AS year,
    i.mes                                            AS month,
    i.puerto_desembarque                             AS port,
    'import'                                         AS direction,
    i.aduana,
    COUNT(*)                                         AS shipment_count,
    COALESCE(SUM(i.cif_us), 0)                       AS total_value_usd,
    NULL::DOUBLE                                     AS total_weight_mt,
    COALESCE(SUM(i.cantidad_mercancia), 0)           AS total_quantity,
    COUNT(DISTINCT i.hs2_capitulo)                   AS commodity_diversity,
    COUNT(DISTINCT i.hs4_partida)                    AS hs4_diversity,
    COUNT(DISTINCT i.pais_origen)                    AS country_diversity,
    COUNT(DISTINCT i.continente_origen)              AS continent_diversity,
    MODE(i.hs2_capitulo)                             AS dominant_hs2,
    MODE(i.tipo_carga)                               AS dominant_cargo_type,
    MODE(i.pais_origen)                              AS dominant_origin_country,
    MODE(i.continente_origen)                        AS dominant_continent,
    SUM(CASE WHEN i.tipo_carga ILIKE '%GENERAL%'    THEN 1 ELSE 0 END) AS cnt_general,
    SUM(CASE WHEN i.tipo_carga ILIKE '%GRANEL%'     THEN 1 ELSE 0 END) AS cnt_bulk,
    SUM(CASE WHEN i.tipo_carga ILIKE '%FRIGORI%'    THEN 1 ELSE 0 END) AS cnt_refrigerated,
    SUM(CASE WHEN i.tipo_carga ILIKE '%CONTENEDOR%' THEN 1 ELSE 0 END) AS cnt_container,
    AVG(i.cif_us)                                    AS avg_value_per_shipment_usd
FROM clean_maritime_imports i
WHERE i.puerto_desembarque IS NOT NULL
  AND i.periodo BETWEEN 2005 AND 2025
GROUP BY i.puerto_desembarque, i.periodo, i.mes, i.aduana

UNION ALL

SELECT
    e.periodo                                        AS year,
    e.mes                                            AS month,
    e.puerto_embarque                                AS port,
    'export'                                         AS direction,
    e.aduana,
    COUNT(*)                                         AS shipment_count,
    COALESCE(SUM(e.fob_us), 0)                       AS total_value_usd,
    COALESCE(SUM(e.peso_bruto_kg) / 1000.0, 0)       AS total_weight_mt,
    COALESCE(SUM(e.cantidad_mercancia), 0)           AS total_quantity,
    COUNT(DISTINCT e.hs2_capitulo)                   AS commodity_diversity,
    COUNT(DISTINCT e.hs4_partida)                    AS hs4_diversity,
    COUNT(DISTINCT e.pais_destino)                   AS country_diversity,
    COUNT(DISTINCT e.continente_destino)             AS continent_diversity,
    MODE(e.hs2_capitulo)                             AS dominant_hs2,
    MODE(e.tipo_carga)                               AS dominant_cargo_type,
    MODE(e.pais_destino)                             AS dominant_origin_country,
    MODE(e.continente_destino)                       AS dominant_continent,
    SUM(CASE WHEN e.tipo_carga ILIKE '%GENERAL%'    THEN 1 ELSE 0 END),
    SUM(CASE WHEN e.tipo_carga ILIKE '%GRANEL%'     THEN 1 ELSE 0 END),
    SUM(CASE WHEN e.tipo_carga ILIKE '%FRIGORI%'    THEN 1 ELSE 0 END),
    SUM(CASE WHEN e.tipo_carga ILIKE '%CONTENEDOR%' THEN 1 ELSE 0 END),
    AVG(e.fob_us)
FROM clean_maritime_exports e
WHERE e.puerto_embarque IS NOT NULL
  AND e.periodo BETWEEN 2005 AND 2025
GROUP BY e.puerto_embarque, e.periodo, e.mes, e.aduana
"""

agg = con.sql(agg_sql).fetchdf()
print(f"  ✓ port_monthly_agg: {len(agg):,} rows  ({time.time()-t0:.1f}s)")

agg_path = os.path.join(OUT_DIR, "port_monthly_agg.parquet")
agg.to_parquet(agg_path, index=False)
print(f"  → {agg_path}")

# ── 2. PORT FEATURES INDEXED ──────────────────────────────────────
print("\nStep 2/2 — Building port_features_indexed ...")
t1 = time.time()

# We register agg into duckdb as an in-memory view, then run the lag/rolling SQL
mem = duckdb.connect()
mem.register("port_monthly_agg", agg)

feat_sql = """
WITH base AS (
    SELECT *,
        CASE
            WHEN month IN (12,1,2) THEN 'Summer'
            WHEN month IN (3,4,5)  THEN 'Autumn'
            WHEN month IN (6,7,8)  THEN 'Winter'
            ELSE 'Spring'
        END                                                              AS season,
        ((month - 1) / 3 + 1)::DOUBLE                                    AS quarter,
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
        AVG(shipment_count::DOUBLE)
            OVER (PARTITION BY port, direction ORDER BY year, month
                  ROWS BETWEEN 2  PRECEDING AND CURRENT ROW)            AS rolling_3_mean,
        AVG(shipment_count::DOUBLE)
            OVER (PARTITION BY port, direction ORDER BY year, month
                  ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)            AS rolling_12_mean,
        AVG(total_value_usd)
            OVER (PARTITION BY port, direction ORDER BY year, month
                  ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)            AS rolling_value_12_mean,
        (shipment_count - LAG(shipment_count, 12) OVER w)::DOUBLE
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
FROM with_norm
"""

feat = mem.sql(feat_sql).fetchdf()
print(f"  ✓ port_features_indexed: {len(feat):,} rows × {len(feat.columns)} cols  "
      f"({time.time()-t1:.1f}s)")

# ── COVID-aware engineered features (mirrors 05_ml_train_evaluate.py) ──
print("  Adding COVID-aware features ...")
df = feat.sort_values(["port", "direction", "year", "month"]).reset_index(drop=True)
df["is_covid_shock"]      = (df["year"] == 2020).astype(np.int8)
df["is_covid_rebound"]    = (df["year"] == 2021).astype(np.int8)
df["is_covid_aftershock"] = (df["year"] == 2022).astype(np.int8)
df["lag_12_is_covid"]     = df["year"].apply(
    lambda y: np.int8(1) if (y - 1) in COVID_YEARS else np.int8(0)
)

# Pre-COVID per-month average per port-direction (used to clean lag_12)
pre_mask = df["year"].isin(range(2015, 2020))
pre = (df[pre_mask]
       .groupby(["port", "direction", "month"])["shipment_count"]
       .mean()
       .rename("pre_covid_monthly_avg")
       .reset_index())
df = df.merge(pre, on=["port", "direction", "month"], how="left")

clean_mask = df["year"].isin(CLEAN_YEARS) & df["yoy_growth"].notna()
cyoy = (df[clean_mask]
        .groupby(["port", "direction"])["yoy_growth"]
        .median()
        .rename("clean_yoy_overall")
        .reset_index())
df = df.merge(cyoy, on=["port", "direction"], how="left")
df["clean_yoy_overall"] = df["clean_yoy_overall"].fillna(0.03)

def _clean_lag12(row):
    if row["lag_12_is_covid"] == 0:
        return row["lag_12"]
    base = row["pre_covid_monthly_avg"]
    if pd.isna(base) or base <= 0:
        base = row["rolling_12_mean"] or 1
    years_out = row["year"] - 1 - 2019
    g = max(-0.15, min(0.15, row["clean_yoy_overall"]))
    return max(1.0, float(base) * ((1 + g) ** years_out))

df["lag_12_clean"] = df.apply(_clean_lag12, axis=1)
df["yoy_growth_clean"] = np.where(
    df["year"].isin(COVID_YEARS),
    df["clean_yoy_overall"],
    df["yoy_growth"].fillna(df["clean_yoy_overall"]),
)
df = df.drop(columns=["pre_covid_monthly_avg", "clean_yoy_overall"], errors="ignore")

feat_path = os.path.join(OUT_DIR, "port_features_indexed.parquet")
df.to_parquet(feat_path, index=False)
print(f"  → {feat_path}")

# ── 3. Tiny meta file with constants ──────────────────────────────
meta = {
    "covid_years":      sorted(COVID_YEARS),
    "incomplete_years": sorted(INCOMPLETE_YEARS),
    "clean_years":      sorted(CLEAN_YEARS),
    "n_port_direction": int(df.groupby(["port", "direction"]).ngroups),
    "n_ports":          int(df["port"].nunique()),
    "n_rows_features":  int(len(df)),
    "n_rows_agg":       int(len(agg)),
    "year_range":       [int(df["year"].min()), int(df["year"].max())],
}
import json
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"  → {os.path.join(OUT_DIR, 'meta.json')}")

print(f"\nDone in {time.time()-t0:.1f}s.")
print(f"  port_monthly_agg      : {len(agg):,} rows")
print(f"  port_features_indexed : {len(df):,} rows × {len(df.columns)} cols")
print(f"  ports                 : {df['port'].nunique()}")
print(f"  port-direction pairs  : {df.groupby(['port','direction']).ngroups}")
