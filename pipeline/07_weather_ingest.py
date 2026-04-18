"""
07_weather_ingest.py
Waze Cargo — Maritime Weather ETL Pipeline

Modes:
  --backfill          Fetch historical data 2002-01-01 → yesterday for all ports
  --forecast          Fetch 7-day forecast and score for all ports (run daily)
  --aggregate         Rebuild port_weather_monthly from hourly data (run monthly)
  --port PORT_KEY     Run only for one port (use with any mode)
  --start-year YYYY   Override backfill start year (default 2002)

Tables written to waze_cargo schema on RDS:
  port_weather_hourly       Raw API values per port per hour
  port_operability_score    Computed operability index + risk label
  port_weather_monthly      Aggregated ML features per port per month

Usage:
  source venv/bin/activate
  export RDS_HOST=wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com
  export RDS_USER=waze_admin
  export RDS_PASSWORD='...'
  export RDS_DBNAME=waze_cargo

  # First run — full historical backfill (takes ~30-60 min for all ports)
  python 07_weather_ingest.py --backfill

  # Daily cron job
  python 07_weather_ingest.py --forecast

  # Rebuild monthly ML features after backfill
  python 07_weather_ingest.py --aggregate

  # Test on one port first
  python 07_weather_ingest.py --backfill --port VALPARAISO --start-year 2020
"""

import os
import sys
import logging
import argparse
import time
from datetime import date, datetime, timedelta

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Local modules (must be in same folder)
from ports_config import PORTS
from weather_api_client import (
    fetch_forecast,
    fetch_historical_chunk,
    chunk_date_range,
    MARINE_ARCHIVE_VARS,
    WIND_ARCHIVE_VARS,
)
from operability_engine import score_dataframe

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("waze_cargo.weather_ingest")

# ── RDS connection ─────────────────────────────────────────────────────────────
PG_SCHEMA = "waze_cargo"

def get_conn():
    return psycopg2.connect(
        host     = os.environ["RDS_HOST"],
        user     = os.environ["RDS_USER"],
        password = os.environ["RDS_PASSWORD"],
        dbname   = os.environ.get("RDS_DBNAME", "waze_cargo"),
        port     = 5432,
        sslmode  = "require",
        keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=3,
    )


# ── Schema creation ────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = f"""
SET search_path TO {PG_SCHEMA};

-- Raw hourly weather per port
CREATE TABLE IF NOT EXISTS port_weather_hourly (
    id                   BIGSERIAL PRIMARY KEY,
    port                 TEXT        NOT NULL,
    zone                 TEXT        NOT NULL,
    ts                   TIMESTAMPTZ NOT NULL,
    wave_height_m        REAL,
    wave_direction_deg   REAL,
    wave_period_s        REAL,
    swell_height_m       REAL,
    swell_direction_deg  REAL,
    swell_period_s       REAL,
    wind_wave_height_m   REAL,
    wind_speed_kmh       REAL,
    wind_direction_deg   REAL,
    wind_gust_kmh        REAL,
    is_forecast          BOOLEAN     NOT NULL DEFAULT FALSE,
    fetched_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_pwh_port_ts
    ON port_weather_hourly(port, ts);
CREATE INDEX IF NOT EXISTS ix_pwh_port_zone
    ON port_weather_hourly(port, zone);

-- Scored operability per port per hour
CREATE TABLE IF NOT EXISTS port_operability_score (
    id                   BIGSERIAL PRIMARY KEY,
    port                 TEXT        NOT NULL,
    zone                 TEXT        NOT NULL,
    ts                   TIMESTAMPTZ NOT NULL,
    wave_height_m        REAL,
    wave_direction_deg   REAL,
    wave_period_s        REAL,
    swell_height_m       REAL,
    swell_direction_deg  REAL,
    swell_period_s       REAL,
    wind_wave_height_m   REAL,
    wind_speed_kmh       REAL,
    wind_direction_deg   REAL,
    wind_gust_kmh        REAL,
    exposure_factor      REAL,
    wave_risk_score      REAL,
    wind_risk_score      REAL,
    operability_index    REAL        NOT NULL,
    risk_level           SMALLINT    NOT NULL,
    risk_label           TEXT        NOT NULL,
    closure_driver       TEXT,
    swell_dominant       BOOLEAN,
    hs_threshold_m       REAL,
    hs_closure_m         REAL,
    is_forecast          BOOLEAN     NOT NULL DEFAULT FALSE,
    scored_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_pos_port_ts
    ON port_operability_score(port, ts);
CREATE INDEX IF NOT EXISTS ix_pos_risk
    ON port_operability_score(port, risk_level, ts);

-- Monthly aggregated ML features
CREATE TABLE IF NOT EXISTS port_weather_monthly (
    port                      TEXT     NOT NULL,
    zone                      TEXT     NOT NULL,
    year                      INTEGER  NOT NULL,
    month                     INTEGER  NOT NULL,
    avg_wave_height_m         REAL,
    max_wave_height_m         REAL,
    avg_swell_height_m        REAL,
    max_swell_height_m        REAL,
    avg_wave_period_s         REAL,
    avg_wind_speed_kmh        REAL,
    max_wind_gust_kmh         REAL,
    avg_operability_index     REAL,
    min_operability_index     REAL,
    pct_hours_normal          REAL,    -- % hours at risk level 0
    pct_hours_watch           REAL,    -- % hours at risk level 1
    pct_hours_advisory        REAL,    -- % hours at risk level 2
    pct_hours_warning         REAL,    -- % hours at risk level 3
    pct_hours_closed          REAL,    -- % hours at risk level 4
    n_closure_events          INTEGER, -- number of distinct closure episodes
    max_consecutive_closure_h INTEGER, -- longest continuous closure in hours
    dominant_closure_driver   TEXT,    -- most common driver when risk >= 3
    dominant_wave_direction   REAL,    -- avg direction in high-risk hours
    swell_dominated_pct       REAL,    -- % high-risk hours where swell dominated
    data_coverage_pct         REAL,    -- % of expected hours with data
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (port, year, month)
);
CREATE INDEX IF NOT EXISTS ix_pwm_port_year
    ON port_weather_monthly(port, year, month);
"""


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLES_SQL)
    conn.commit()
    log.info("Schema verified — 3 weather tables ready")


# ── Helpers ────────────────────────────────────────────────────────────────────

def api_response_to_df(data: dict, port_key: str) -> pd.DataFrame:
    """Convert Open-Meteo JSON response to a tidy DataFrame."""
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"ts": pd.to_datetime(times, utc=False)})
    df["ts"] = df["ts"].dt.tz_localize("America/Santiago", ambiguous="NaT", nonexistent="NaT")
    df = df[df["ts"].notna()].reset_index(drop=True)
    df = df.dropna(subset=["ts"]).reset_index(drop=True)

    col_map = {
        "wave_height":         "wave_height_m",
        "wave_direction":      "wave_direction_deg",
        "wave_period":         "wave_period_s",
        "swell_wave_height":   "swell_height_m",
        "swell_wave_direction":"swell_direction_deg",
        "swell_wave_period":   "swell_period_s",
        "wind_wave_height":    "wind_wave_height_m",
        "wind_speed_10m":      "wind_speed_kmh",
        "wind_direction_10m":  "wind_direction_deg",
        "wind_gusts_10m":      "wind_gust_kmh",
    }
    for api_col, db_col in col_map.items():
        if api_col in hourly:
            df[api_col] = [hourly[api_col][i] for i in df.index]  # aligned to filtered index
        else:
            df[api_col] = None

    df["port"] = port_key
    df["zone"] = PORTS[port_key]["zone"]
    return df


def get_last_loaded_ts(conn, port: str) -> date:
    """Return the date of the most recent row in port_weather_hourly for this port."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MAX(ts)::date FROM {PG_SCHEMA}.port_weather_hourly WHERE port = %s",
            (port,)
        )
        result = cur.fetchone()[0]
    return result if result else date(2002, 1, 1)


def upsert_hourly(conn, rows: list[dict], is_forecast: bool = False):
    """Upsert rows into port_weather_hourly."""
    if not rows:
        return
    cols = [
        "port","zone","ts","wave_height_m","wave_direction_deg","wave_period_s",
        "swell_height_m","swell_direction_deg","swell_period_s","wind_wave_height_m",
        "wind_speed_kmh","wind_direction_deg","wind_gust_kmh","is_forecast",
    ]
    values = []
    for r in rows:
        values.append((
            r["port"], r["zone"], r["ts"],
            r.get("wave_height"), r.get("wave_direction"), r.get("wave_period"),
            r.get("swell_wave_height"), r.get("swell_wave_direction"), r.get("swell_wave_period"),
            r.get("wind_wave_height"),
            r.get("wind_speed_10m"), r.get("wind_direction_10m"), r.get("wind_gusts_10m"),
            is_forecast,
        ))
    sql = f"""
        INSERT INTO {PG_SCHEMA}.port_weather_hourly ({','.join(cols)})
        VALUES %s
        ON CONFLICT (port, ts) DO UPDATE SET
            wave_height_m      = EXCLUDED.wave_height_m,
            swell_height_m     = EXCLUDED.swell_height_m,
            wind_speed_kmh     = EXCLUDED.wind_speed_kmh,
            wind_gust_kmh      = EXCLUDED.wind_gust_kmh,
            is_forecast        = EXCLUDED.is_forecast,
            fetched_at         = NOW()
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=5000)
    conn.commit()


def upsert_scores(conn, df_scored: pd.DataFrame, is_forecast: bool = False):
    """Upsert scored operability rows."""
    if df_scored.empty:
        return
    cols = [
        "port","zone","ts","wave_height_m","wave_direction_deg","wave_period_s",
        "swell_height_m","swell_direction_deg","swell_period_s","wind_wave_height_m",
        "wind_speed_kmh","wind_direction_deg","wind_gust_kmh",
        "exposure_factor","wave_risk_score","wind_risk_score",
        "operability_index","risk_level","risk_label","closure_driver",
        "swell_dominant","hs_threshold_m","hs_closure_m","is_forecast",
    ]
    values = []
    for _, row in df_scored.iterrows():
        values.append(tuple(
            row.get(c) if c != "is_forecast" else is_forecast
            for c in cols
        ))
    sql = f"""
        INSERT INTO {PG_SCHEMA}.port_operability_score ({','.join(cols)})
        VALUES %s
        ON CONFLICT (port, ts) DO UPDATE SET
            operability_index = EXCLUDED.operability_index,
            risk_level        = EXCLUDED.risk_level,
            risk_label        = EXCLUDED.risk_label,
            closure_driver    = EXCLUDED.closure_driver,
            is_forecast       = EXCLUDED.is_forecast,
            scored_at         = NOW()
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=5000)
    conn.commit()


# ── Modes ──────────────────────────────────────────────────────────────────────

def run_backfill(conn, ports: list[str], start_year: int = 2002):
    """Fetch full historical data 2002→yesterday for all ports."""
    start = date(start_year, 1, 1)
    end   = date.today() - timedelta(days=1)

    log.info("=" * 70)
    log.info("BACKFILL mode: %s → %s | %d ports", start, end, len(ports))
    log.info("=" * 70)

    for port_key in ports:
        cfg = PORTS[port_key]
        log.info("\n── %s (%s zone) ──", port_key, cfg["zone"])

        # Find where we left off (resume support)
        last_loaded = get_last_loaded_ts(conn, port_key)
        if last_loaded >= end:
            log.info("  Already up to date (%s) — skipping", last_loaded)
            continue
        chunk_start = last_loaded + timedelta(days=1) if last_loaded > date(2002, 1, 1) else start

        chunks = chunk_date_range(chunk_start, end, months=12)
        log.info("  Fetching %d annual chunks from %s", len(chunks), chunk_start)

        total_rows = 0
        for chunk_s, chunk_e in chunks:
            log.info("    Chunk %s → %s", chunk_s, chunk_e)
            try:
                data = fetch_historical_chunk(cfg["lat"], cfg["lon"], chunk_s, chunk_e)
                df   = api_response_to_df(data, port_key)
                if df.empty:
                    log.warning("    Empty response for %s %s→%s", port_key, chunk_s, chunk_e)
                    continue

                # Store raw hourly
                upsert_hourly(conn, df.to_dict("records"), is_forecast=False)

                # Score and store operability
                df_scored = score_dataframe(port_key, df)
                df_scored["ts"]   = df["ts"].values
                df_scored["port"] = port_key
                df_scored["zone"] = cfg["zone"]
                upsert_scores(conn, df_scored, is_forecast=False)

                total_rows += len(df)
                log.info("    ✓ %d rows", len(df))
                time.sleep(0.5)   # gentle rate limiting

            except Exception as e:
                log.error("    ERROR on %s chunk %s→%s: %s", port_key, chunk_s, chunk_e, e)
                time.sleep(5)
                continue

        log.info("  ✓ %s complete — %d total rows", port_key, total_rows)

    log.info("\nBackfill complete.")


def run_forecast(conn, ports: list[str]):
    """Fetch 7-day forecast and score for all ports."""
    log.info("=" * 70)
    log.info("FORECAST mode — 7-day outlook | %d ports", len(ports))
    log.info("=" * 70)

    for port_key in ports:
        cfg = PORTS[port_key]
        try:
            data = fetch_forecast(cfg["lat"], cfg["lon"])
            df   = api_response_to_df(data, port_key)
            if df.empty:
                continue

            upsert_hourly(conn, df.to_dict("records"), is_forecast=True)

            df_scored = score_dataframe(port_key, df)
            df_scored["ts"]   = df["ts"].values
            df_scored["port"] = port_key
            df_scored["zone"] = cfg["zone"]
            upsert_scores(conn, df_scored, is_forecast=True)

            # Print high-risk alerts
            alerts = df_scored[df_scored["risk_level"] >= 3]
            if not alerts.empty:
                log.warning("  ⚠  %s: %d WARNING/CLOSED hours in next 7 days",
                            port_key, len(alerts))
                for _, a in alerts.head(3).iterrows():
                    log.warning("     %s  risk=%s  driver=%s  Hs=%.2fm",
                                a["ts"], a["risk_label"], a["closure_driver"],
                                a.get("wave_height_m") or 0)
            else:
                log.info("  ✓ %s — no high-risk periods in 7-day window", port_key)

            time.sleep(0.3)

        except Exception as e:
            log.error("  ERROR on %s: %s", port_key, e)

    log.info("Forecast run complete.")


def run_aggregate(conn):
    """Rebuild port_weather_monthly from hourly operability scores."""
    log.info("Building port_weather_monthly ...")

    sql = f"""
    INSERT INTO {PG_SCHEMA}.port_weather_monthly (
        port, zone, year, month,
        avg_wave_height_m, max_wave_height_m,
        avg_swell_height_m, max_swell_height_m,
        avg_wave_period_s, avg_wind_speed_kmh, max_wind_gust_kmh,
        avg_operability_index, min_operability_index,
        pct_hours_normal, pct_hours_watch, pct_hours_advisory,
        pct_hours_warning, pct_hours_closed,
        n_closure_events, max_consecutive_closure_h,
        dominant_closure_driver, dominant_wave_direction,
        swell_dominated_pct, data_coverage_pct, updated_at
    )
    SELECT
        port,
        zone,
        EXTRACT(YEAR  FROM ts)::INTEGER AS year,
        EXTRACT(MONTH FROM ts)::INTEGER AS month,
        ROUND(AVG(wave_height_m)::NUMERIC,    3) AS avg_wave_height_m,
        ROUND(MAX(wave_height_m)::NUMERIC,    3) AS max_wave_height_m,
        ROUND(AVG(swell_height_m)::NUMERIC,   3) AS avg_swell_height_m,
        ROUND(MAX(swell_height_m)::NUMERIC,   3) AS max_swell_height_m,
        ROUND(AVG(wave_period_s)::NUMERIC,    2) AS avg_wave_period_s,
        ROUND(AVG(wind_speed_kmh)::NUMERIC,   2) AS avg_wind_speed_kmh,
        ROUND(MAX(wind_gust_kmh)::NUMERIC,    2) AS max_wind_gust_kmh,
        ROUND(AVG(operability_index)::NUMERIC,4) AS avg_operability_index,
        ROUND(MIN(operability_index)::NUMERIC,4) AS min_operability_index,
        ROUND(100.0 * SUM(CASE WHEN risk_level = 0 THEN 1 ELSE 0 END) / COUNT(*), 2),
        ROUND(100.0 * SUM(CASE WHEN risk_level = 1 THEN 1 ELSE 0 END) / COUNT(*), 2),
        ROUND(100.0 * SUM(CASE WHEN risk_level = 2 THEN 1 ELSE 0 END) / COUNT(*), 2),
        ROUND(100.0 * SUM(CASE WHEN risk_level = 3 THEN 1 ELSE 0 END) / COUNT(*), 2),
        ROUND(100.0 * SUM(CASE WHEN risk_level = 4 THEN 1 ELSE 0 END) / COUNT(*), 2),
        -- Count distinct closure events (transitions from <4 to 4)
        (SELECT COUNT(*) FROM (
            SELECT port AS p, ts AS t,
                   risk_level,
                   LAG(risk_level) OVER (PARTITION BY port ORDER BY ts) AS prev_rl
            FROM {PG_SCHEMA}.port_operability_score s2
            WHERE s2.port = s.port
              AND EXTRACT(YEAR FROM s2.ts)  = EXTRACT(YEAR FROM s.ts)
              AND EXTRACT(MONTH FROM s2.ts) = EXTRACT(MONTH FROM s.ts)
        ) ev WHERE risk_level = 4 AND (prev_rl IS NULL OR prev_rl < 4)),
        NULL,   -- max_consecutive_closure_h (computed separately if needed)
        -- Most common closure driver when risk >= 3
        (SELECT closure_driver FROM {PG_SCHEMA}.port_operability_score s3
         WHERE s3.port = s.port AND risk_level >= 3
           AND EXTRACT(YEAR FROM s3.ts)  = EXTRACT(YEAR FROM s.ts)
           AND EXTRACT(MONTH FROM s3.ts) = EXTRACT(MONTH FROM s.ts)
         GROUP BY closure_driver ORDER BY COUNT(*) DESC LIMIT 1),
        ROUND(AVG(CASE WHEN risk_level >= 3 THEN wave_direction_deg END)::NUMERIC, 1),
        ROUND(100.0 * SUM(CASE WHEN swell_dominant THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2),
        -- Data coverage (% of 730 expected hours/month)
        ROUND(100.0 * COUNT(*) / 730.0, 2),
        NOW()
    FROM {PG_SCHEMA}.port_operability_score s
    WHERE is_forecast = FALSE
    GROUP BY port, zone, EXTRACT(YEAR FROM ts), EXTRACT(MONTH FROM ts)
    ON CONFLICT (port, year, month) DO UPDATE SET
        avg_wave_height_m       = EXCLUDED.avg_wave_height_m,
        max_wave_height_m       = EXCLUDED.max_wave_height_m,
        avg_swell_height_m      = EXCLUDED.avg_swell_height_m,
        avg_operability_index   = EXCLUDED.avg_operability_index,
        pct_hours_closed        = EXCLUDED.pct_hours_closed,
        n_closure_events        = EXCLUDED.n_closure_events,
        data_coverage_pct       = EXCLUDED.data_coverage_pct,
        updated_at              = NOW();
    """

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {PG_SCHEMA}.port_weather_monthly")
        n = cur.fetchone()[0]
    log.info("  ✓ port_weather_monthly: %d rows", n)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Waze Cargo — Maritime Weather ETL")
    parser.add_argument("--backfill",    action="store_true", help="Fetch full historical data")
    parser.add_argument("--forecast",    action="store_true", help="Fetch 7-day forecast")
    parser.add_argument("--aggregate",   action="store_true", help="Rebuild monthly ML features")
    parser.add_argument("--port",        type=str,            help="Run only for this port key")
    parser.add_argument("--start-year",  type=int, default=2002, help="Backfill start year")
    args = parser.parse_args()

    if not any([args.backfill, args.forecast, args.aggregate]):
        parser.print_help()
        sys.exit(1)

    ports = [args.port.upper()] if args.port else list(PORTS.keys())

    log.info("Waze Cargo — Maritime Weather Ingest")
    log.info("Ports: %s", ports)

    conn = get_conn()
    ensure_schema(conn)

    if args.backfill:
        run_backfill(conn, ports, start_year=args.start_year)

    if args.forecast:
        run_forecast(conn, ports)

    if args.aggregate:
        run_aggregate(conn)

    conn.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
