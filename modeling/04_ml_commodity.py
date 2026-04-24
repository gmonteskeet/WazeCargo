"""
WAZE CARGO — Step 3b: Commodity Port Forecasting ML (RDS PostgreSQL)
=====================================================================
Reads from:   RDS  waze_cargo.clean_maritime_exports
Writes to:    RDS  waze_cargo.commodity_port_params
                   waze_cargo.commodity_seasonal_index
                   waze_cargo.commodity_port_forecast

Exact column names match your MotherDuck schema.

RUN:
    python 04_ml_commodity.py
"""

import logging, os, sys, time
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ml_commodity")
S = "waze_cargo"


def get_engine():
    h,p,u,pw,db = (os.environ.get(k,"") for k in
                   ["RDS_HOST","RDS_PORT","RDS_USER","RDS_PASSWORD","RDS_DBNAME"])
    p = p or "5432"
    missing = [k for k,v in zip(["RDS_HOST","RDS_USER","RDS_PASSWORD","RDS_DBNAME"],
                                  [h,u,pw,db]) if not v]
    if missing:
        log.error("Missing: %s", " ".join(missing)); sys.exit(1)
    engine = create_engine(
        f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{db}?sslmode=require",
        pool_pre_ping=True, connect_args={"connect_timeout":15})
    with engine.connect() as c: c.execute(text("SELECT 1"))
    log.info("Connected → %s/%s", h, db)
    return engine


def run(engine, label, sql):
    t0 = time.time()
    with engine.begin() as c:
        c.execute(text(sql))
    tbl = label.split("→")[-1].strip()
    try:
        with engine.connect() as c:
            n = c.execute(text(f"SELECT COUNT(*) FROM {S}.{tbl}")).scalar()
        log.info("  ✓ %-40s  %9d rows  %.1fs", label, n, time.time()-t0)
    except Exception:
        log.info("  ✓ %-40s  %.1fs", label, time.time()-t0)


# ════════════════════════════════════════════════════════════════
#  1 — COMMODITY PORT PARAMS
#  Matches DuckDB schema: hs2, port, n_obs, last_year, last_month,
#  avg_monthly_fob, avg_monthly_weight_mt, avg_monthly_ships,
#  avg_yoy_fob_growth, avg_yoy_weight_growth, avg_yoy_ships_growth,
#  dominant_description
# ════════════════════════════════════════════════════════════════

SQL_PARAMS = f"""
DROP TABLE IF EXISTS {S}.commodity_port_params CASCADE;
CREATE TABLE {S}.commodity_port_params AS
WITH base AS (
    SELECT
        hs2_capitulo                                AS hs2,
        puerto_embarque                             AS port,
        periodo                                     AS year,
        mes                                         AS month,
        SUM(fob_us)                                 AS fob_usd,
        SUM(peso_bruto_kg) / 1000.0                 AS weight_mt,
        COUNT(*)                                    AS shipment_count,
        MODE() WITHIN GROUP (ORDER BY descripcion_producto) AS top_description
    FROM {S}.clean_maritime_exports
    WHERE hs2_capitulo IS NOT NULL
      AND puerto_embarque IS NOT NULL
      AND periodo BETWEEN 2010 AND 2025
    GROUP BY hs2_capitulo, puerto_embarque, periodo, mes
),
last_p AS (
    SELECT hs2, port,
        (MAX(year * 100 + month) / 100)::INTEGER AS last_year,
        (MAX(year * 100 + month) % 100)::INTEGER AS last_month
    FROM base GROUP BY hs2, port
),
with_yoy AS (
    SELECT b.*,
        LAG(fob_usd,       12) OVER (PARTITION BY b.hs2, b.port ORDER BY b.year, b.month) AS lag_fob_12,
        LAG(weight_mt,     12) OVER (PARTITION BY b.hs2, b.port ORDER BY b.year, b.month) AS lag_weight_12,
        LAG(shipment_count,12) OVER (PARTITION BY b.hs2, b.port ORDER BY b.year, b.month) AS lag_ships_12
    FROM base b
),
stats AS (
    SELECT
        w.hs2, w.port,
        COUNT(*)::BIGINT                            AS n_obs,
        l.last_year, l.last_month,
        AVG(w.fob_usd)                              AS avg_monthly_fob,
        AVG(w.weight_mt)                            AS avg_monthly_weight_mt,
        AVG(w.shipment_count::DOUBLE PRECISION)     AS avg_monthly_ships,
        AVG(CASE WHEN w.lag_fob_12 > 0
                 THEN (w.fob_usd - w.lag_fob_12) / w.lag_fob_12 END)       AS avg_yoy_fob_growth,
        AVG(CASE WHEN w.lag_weight_12 > 0
                 THEN (w.weight_mt - w.lag_weight_12) / w.lag_weight_12 END) AS avg_yoy_weight_growth,
        AVG(CASE WHEN w.lag_ships_12 > 0
                 THEN (w.shipment_count - w.lag_ships_12)::DOUBLE PRECISION
                      / w.lag_ships_12 END)                                 AS avg_yoy_ships_growth,
        MODE() WITHIN GROUP (ORDER BY w.top_description) AS dominant_description
    FROM with_yoy w
    JOIN last_p l ON w.hs2 = l.hs2 AND w.port = l.port
    WHERE w.lag_fob_12 IS NOT NULL
    GROUP BY w.hs2, w.port, l.last_year, l.last_month
    HAVING COUNT(*) >= 48
)
SELECT * FROM stats
ORDER BY avg_monthly_fob * n_obs DESC;
"""

# ════════════════════════════════════════════════════════════════
#  2 — COMMODITY SEASONAL INDEX
#  Matches DuckDB schema: hs2, port, month, avg_monthly_fob,
#  seasonal_fob_factor, seasonal_weight_factor, overall_avg_ships
# ════════════════════════════════════════════════════════════════

SQL_SEASONAL = f"""
DROP TABLE IF EXISTS {S}.commodity_seasonal_index CASCADE;
CREATE TABLE {S}.commodity_seasonal_index AS
SELECT
    hs2_capitulo                                                    AS hs2,
    puerto_embarque                                                 AS port,
    mes                                                             AS month,
    AVG(fob_us)                                                     AS avg_monthly_fob,
    AVG(fob_us)
        / NULLIF(AVG(AVG(fob_us))
                 OVER (PARTITION BY hs2_capitulo, puerto_embarque), 0) AS seasonal_fob_factor,
    AVG(COALESCE(peso_bruto_kg,0)/1000.0)
        / NULLIF(AVG(AVG(COALESCE(peso_bruto_kg,0)/1000.0))
                 OVER (PARTITION BY hs2_capitulo, puerto_embarque), 0) AS seasonal_weight_factor,
    AVG(AVG(COUNT(*)) OVER ())                                      AS overall_avg_ships
FROM {S}.clean_maritime_exports
WHERE hs2_capitulo IS NOT NULL
  AND puerto_embarque IS NOT NULL
  AND periodo BETWEEN 2015 AND 2025
GROUP BY hs2_capitulo, puerto_embarque, mes
HAVING COUNT(DISTINCT periodo) >= 4;
"""

# ════════════════════════════════════════════════════════════════
#  2b — Fix overall_avg_ships (the nested aggregate trick above
#        doesn't work cleanly in PostgreSQL — use a simpler approach)
# ════════════════════════════════════════════════════════════════

SQL_SEASONAL_FIXED = f"""
DROP TABLE IF EXISTS {S}.commodity_seasonal_index CASCADE;
CREATE TABLE {S}.commodity_seasonal_index AS
WITH monthly AS (
    SELECT
        hs2_capitulo                            AS hs2,
        puerto_embarque                         AS port,
        mes                                     AS month,
        AVG(fob_us)                             AS avg_monthly_fob,
        AVG(COALESCE(peso_bruto_kg,0)/1000.0)  AS avg_monthly_wt,
        COUNT(DISTINCT periodo)                 AS years_active
    FROM {S}.clean_maritime_exports
    WHERE hs2_capitulo IS NOT NULL
      AND puerto_embarque IS NOT NULL
      AND periodo BETWEEN 2015 AND 2025
    GROUP BY hs2_capitulo, puerto_embarque, mes
    HAVING COUNT(DISTINCT periodo) >= 4
),
overall AS (
    SELECT hs2, port,
        AVG(avg_monthly_fob)  AS mean_fob,
        AVG(avg_monthly_wt)   AS mean_wt,
        SUM(years_active)*1.0 / 12 AS overall_avg_ships
    FROM monthly
    GROUP BY hs2, port
)
SELECT
    m.hs2, m.port, m.month,
    m.avg_monthly_fob,
    m.avg_monthly_fob / NULLIF(o.mean_fob, 0)  AS seasonal_fob_factor,
    m.avg_monthly_wt  / NULLIF(o.mean_wt,  0)  AS seasonal_weight_factor,
    o.overall_avg_ships
FROM monthly m
JOIN overall o ON m.hs2 = o.hs2 AND m.port = o.port;
"""

# ════════════════════════════════════════════════════════════════
#  3 — COMMODITY PORT FORECAST
#  Matches DuckDB schema: hs2_capitulo, puerto_embarque, year, month,
#  season, pred_fob_usd, pred_weight_mt, pred_shipment_count,
#  season_activity_index, season_label, commodity_description,
#  historical_months, model_version, forecast_generated_at
# ════════════════════════════════════════════════════════════════

SQL_FORECAST = f"""
DROP TABLE IF EXISTS {S}.commodity_port_forecast CASCADE;
CREATE TABLE {S}.commodity_port_forecast AS
WITH offsets AS (SELECT generate_series(1,12) AS step),
grid AS (
    SELECT p.*, o.step,
        CASE WHEN (p.last_month + o.step) > 12
             THEN p.last_year + ((p.last_month + o.step - 1) / 12)
             ELSE p.last_year
        END::INTEGER AS forecast_year,
        (((p.last_month + o.step - 1) % 12) + 1)::INTEGER AS forecast_month
    FROM {S}.commodity_port_params p CROSS JOIN offsets o
),
with_s AS (
    SELECT g.*,
        LEAST(3.0, GREATEST(0.1, COALESCE(s.seasonal_fob_factor,    1.0))) AS sfob,
        LEAST(3.0, GREATEST(0.1, COALESCE(s.seasonal_weight_factor, 1.0))) AS swt
    FROM grid g
    LEFT JOIN {S}.commodity_seasonal_index s
           ON s.hs2 = g.hs2 AND s.port = g.port AND s.month = g.forecast_month
),
computed AS (
    SELECT *,
        POWER(1 + GREATEST(-0.30, LEAST(0.40,
              COALESCE(NULLIF(avg_yoy_fob_growth,   0), 0.025))), step/12.0) AS gf_fob,
        POWER(1 + GREATEST(-0.30, LEAST(0.40,
              COALESCE(NULLIF(avg_yoy_weight_growth,0), 0.025))), step/12.0) AS gf_wt,
        POWER(1 + GREATEST(-0.30, LEAST(0.40,
              COALESCE(NULLIF(avg_yoy_ships_growth, 0), 0.025))), step/12.0) AS gf_sh
    FROM with_s
)
SELECT
    hs2                                                             AS hs2_capitulo,
    port                                                            AS puerto_embarque,
    forecast_year                                                   AS year,
    forecast_month                                                  AS month,
    CASE WHEN forecast_month IN (12,1,2) THEN 'Summer'
         WHEN forecast_month IN (3,4,5)  THEN 'Autumn'
         WHEN forecast_month IN (6,7,8)  THEN 'Winter'
         ELSE 'Spring' END                                          AS season,
    ROUND((avg_monthly_fob * sfob * gf_fob)::NUMERIC, 2)::DOUBLE PRECISION
                                                                    AS pred_fob_usd,
    ROUND((avg_monthly_weight_mt * swt * gf_wt)::NUMERIC, 2)::DOUBLE PRECISION
                                                                    AS pred_weight_mt,
    GREATEST(1, ROUND(avg_monthly_ships * sfob * gf_sh))::INTEGER   AS pred_shipment_count,
    ROUND(LEAST(1.0, GREATEST(0.0, (sfob - 0.1) / 2.9))::NUMERIC, 4)::DOUBLE PRECISION
                                                                    AS season_activity_index,
    CASE WHEN sfob >= 1.30 THEN 'High Season'
         WHEN sfob >= 0.90 THEN 'Normal'
         ELSE 'Low Season' END                                      AS season_label,
    dominant_description                                            AS commodity_description,
    n_obs                                                           AS historical_months,
    'v1.0-commodity-rds'                                            AS model_version,
    NOW()                                                           AS forecast_generated_at
FROM computed
ORDER BY hs2_capitulo, puerto_embarque, year, month;
"""


def main():
    engine = get_engine()
    steps  = [
        ("commodity_port_params",      SQL_PARAMS),
        ("commodity_seasonal_index",   SQL_SEASONAL_FIXED),
        ("commodity_port_forecast",    SQL_FORECAST),
    ]
    log.info("═"*60)
    log.info("  WAZE CARGO — Commodity ML  (RDS PostgreSQL)")
    log.info("═"*60)
    for label, sql in steps:
        log.info("Building → %s", label)
        run(engine, f"built → {label}", sql)

    with engine.connect() as c:
        rows = c.execute(text(
            f"SELECT season_label, COUNT(*) n "
            f"FROM {S}.commodity_port_forecast "
            f"GROUP BY 1 ORDER BY 2 DESC"
        )).fetchall()
    log.info("")
    log.info("  commodity_port_forecast season labels:")
    for r in rows:
        log.info("    %-15s  %d", r[0], r[1])
    engine.dispose()
    log.info("Commodity ML complete.")


if __name__ == "__main__":
    main()
