"""
WAZE CARGO — Step 2: Stream waze_cargo.duckdb → AWS RDS PostgreSQL
====================================================================
Reads directly from waze_cargo.duckdb — no raw CSV reprocessing.
All fixes from the migration sessions included:
  ✓ Fresh connection per chunk  (fixes SSL EOF timeout after 45 min)
  ✓ Chunk size 20K rows         (avoids 30s-per-chunk timeout)
  ✓ 3 retries with 5s sleep     (handles transient network blips)
  ✓ TCP keepalives on connect   (prevents idle connection drops)
  ✓ clean_df()                  (converts Int16/Int32/Int64 + NaN → None)
  ✓ NOT NULL dropped on periodo/mes (754K rows legitimately NULL)
  ✓ All text columns → TEXT     (no VARCHAR length truncation)
  ✓ HUGEINT → BIGINT            (PostgreSQL has no HUGEINT)
  ✓ 'id' skipped on core tables (BIGSERIAL in schema)
  ✓ --resume flag               (continues after crash at exact row)

INSTALL (inside venv):
  pip install -r requirements.txt

SET CREDENTIALS (single quotes protect special chars like !):
  export RDS_HOST=wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com
  export RDS_USER=waze_admin
  export RDS_PASSWORD='-C1!m?yODIsskBwbV__25eaPfi5e'
  export RDS_DBNAME=waze_cargo

RUN:
  python 02_duckdb_to_rds.py                    # migrate everything
  python 02_duckdb_to_rds.py --only lookups     # lookup tables only (fast)
  python 02_duckdb_to_rds.py --only core        # imports + exports (~30 min)
  python 02_duckdb_to_rds.py --only ml          # ML tables only (small)
  python 02_duckdb_to_rds.py --only core --resume  # continue after crash
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("duck2rds")

# ── Config ────────────────────────────────────────────────────────
DUCKDB_FILE = Path("waze_cargo.duckdb")
PG_SCHEMA   = "waze_cargo"
CHUNK       = 20_000     # rows per INSERT batch — small enough to avoid SSL timeouts
RETRIES     = 3          # retry attempts per chunk on network error
RETRY_WAIT  = 5          # seconds between retries

# ── DuckDB type → PostgreSQL type ─────────────────────────────────
# All text → TEXT  (avoids every VARCHAR length issue encountered)
# HUGEINT  → BIGINT (PostgreSQL has no HUGEINT)
TYPE_MAP = {
    "BIGINT":                   "BIGINT",
    "INTEGER":                  "INTEGER",
    "SMALLINT":                 "SMALLINT",
    "HUGEINT":                  "BIGINT",
    "DOUBLE":                   "DOUBLE PRECISION",
    "FLOAT":                    "REAL",
    "BOOLEAN":                  "BOOLEAN",
    "VARCHAR":                  "TEXT",
    "DATE":                     "DATE",
    "TIMESTAMP":                "TIMESTAMP",
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ",
    "INT16":                    "SMALLINT",
    "INT32":                    "INTEGER",
    "INT64":                    "BIGINT",
}

# ── Table groups ───────────────────────────────────────────────────
LOOKUP_TABLES = [
    "lkp_aduanas",
    "lkp_clausulas",
    "lkp_harmonized_system",
    "lkp_modalidades_venta",
    "lkp_moneda",
    "lkp_paises",
    "lkp_puertos",
    "lkp_regimen_importacion",
    "lkp_regiones",
    "lkp_tipos_carga",
    "lkp_tipos_operacion",
    "lkp_unidades_medida",
    "lkp_vias_transporte",
]

CORE_TABLES = [
    "clean_maritime_imports",
    "clean_maritime_exports",
]

ML_TABLES = [
    "port_monthly_agg",
    "port_features_indexed",
    "port_seasonal_index",
    "port_forecast_params",
    "port_congestion_forecast",
    "commodity_port_params",
    "commodity_seasonal_index",
    "commodity_port_forecast",
]


# ════════════════════════════════════════════════════════════════
#  CONNECTIONS
# ════════════════════════════════════════════════════════════════

def open_duckdb():
    if not DUCKDB_FILE.exists():
        log.error("waze_cargo.duckdb not found in: %s", Path.cwd())
        log.error("Run this script FROM ~/Waze_Cargo:  cd ~/Waze_Cargo")
        sys.exit(1)
    con = duckdb.connect(str(DUCKDB_FILE), read_only=True)
    log.info("DuckDB: %s  (%.1f GB)",
             DUCKDB_FILE.resolve(),
             DUCKDB_FILE.stat().st_size / 1e9)
    return con


def open_rds():
    h  = os.environ.get("RDS_HOST",     "")
    p  = os.environ.get("RDS_PORT",     "5432")
    u  = os.environ.get("RDS_USER",     "")
    pw = os.environ.get("RDS_PASSWORD", "")
    db = os.environ.get("RDS_DBNAME",   "")

    missing = [k for k, v in [("RDS_HOST", h), ("RDS_USER", u),
                                ("RDS_PASSWORD", pw), ("RDS_DBNAME", db)] if not v]
    if missing:
        log.error("Missing env vars: %s", "  ".join(missing))
        log.error("export RDS_HOST=...  export RDS_USER=...  etc.")
        sys.exit(1)

    # keepalives prevent AWS RDS from dropping idle TCP connections mid-transfer
    url = (
        f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{db}"
        f"?sslmode=require"
        f"&keepalives=1&keepalives_idle=30"
        f"&keepalives_interval=10&keepalives_count=5"
    )
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "connect_timeout":     15,
            "keepalives":          1,
            "keepalives_idle":     30,
            "keepalives_interval": 10,
            "keepalives_count":    5,
        },
    )
    with engine.connect() as c:
        c.execute(text("SELECT 1"))
    log.info("RDS:    %s:%s/%s", h, p, db)
    return engine


# ════════════════════════════════════════════════════════════════
#  AUTO-CREATE TABLE FROM DUCKDB SCHEMA
#  (only used for tables not already defined in 01_rds_schema.sql)
# ════════════════════════════════════════════════════════════════

def ensure_table(duck, engine, table):
    """
    CREATE TABLE IF NOT EXISTS on RDS matching DuckDB column list.
    Uses TYPE_MAP — all text → TEXT, HUGEINT → BIGINT.
    Skips 'id' on core tables (BIGSERIAL defined in schema).
    """
    cols_df = duck.execute(
        "SELECT column_name, data_type "
        "FROM information_schema.columns "
        f"WHERE table_name = '{table}' "
        "ORDER BY ordinal_position"
    ).df()

    col_defs = []
    for _, row in cols_df.iterrows():
        col    = row["column_name"]
        if col == "id" and table in CORE_TABLES:
            continue   # BIGSERIAL already defined in schema
        pg_type = TYPE_MAP.get(row["data_type"].upper(), "TEXT")
        col_defs.append(f'  "{col}" {pg_type}')

    ddl = (
        f"CREATE TABLE IF NOT EXISTS {PG_SCHEMA}.{table} (\n"
        + ",\n".join(col_defs)
        + "\n);"
    )
    with engine.begin() as c:
        c.execute(text(ddl))


# ════════════════════════════════════════════════════════════════
#  DATAFRAME CLEANING
#  Converts all pandas NA-like values → Python None for psycopg2
# ════════════════════════════════════════════════════════════════

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-step cleaning so psycopg2 never sees NaN, pd.NA, NaT, or
    pandas nullable integer types (Int16, Int32, Int64).

    Step 1: convert nullable integer dtypes → object
    Step 2: convert entire frame → object dtype
    Step 3: replace all remaining NA values → Python None
    """
    # Step 1 — nullable integer types first (must precede astype(object))
    nullable_int = ["Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32","UInt64"]
    for col in df.columns:
        if str(df[col].dtype) in nullable_int:
            df[col] = df[col].astype(object).where(df[col].notna(), None)

    # Step 2 + 3
    df = df.astype(object).where(pd.notnull(df), None)
    return df


# ════════════════════════════════════════════════════════════════
#  INSERT ONE CHUNK  (fresh connection + retry)
# ════════════════════════════════════════════════════════════════

def insert_chunk(engine, table, cols_sql, values):
    """
    Opens a FRESH raw psycopg2 connection for every chunk.
    This is the key fix for the SSL EOF timeout — each 20K-row
    batch is completely independent, so a 3-hour migration never
    has a single connection open for more than ~6 seconds.
    """
    sql = (
        f"INSERT INTO {PG_SCHEMA}.{table} ({cols_sql}) "
        f"VALUES %s ON CONFLICT DO NOTHING"
    )
    for attempt in range(1, RETRIES + 1):
        conn = None
        try:
            conn = engine.raw_connection()
            with conn.cursor() as cur:
                execute_values(cur, sql, values, page_size=5_000)
            conn.commit()
            conn.close()
            return
        except Exception as exc:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            if attempt < RETRIES:
                log.warning("    chunk failed (attempt %d/%d): %s — retry in %ds",
                            attempt, RETRIES, exc, RETRY_WAIT)
                time.sleep(RETRY_WAIT)
            else:
                raise


# ════════════════════════════════════════════════════════════════
#  MIGRATE ONE TABLE
# ════════════════════════════════════════════════════════════════

def migrate_table(duck, engine, table, resume=False):
    t_start = time.time()

    total = duck.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if total == 0:
        log.info("  %-42s  empty — skipped", table)
        return 0

    # Resume: check how many rows already on RDS
    rds_count = 0
    if resume:
        try:
            with engine.connect() as c:
                rds_count = c.execute(
                    text(f"SELECT COUNT(*) FROM {PG_SCHEMA}.{table}")
                ).scalar() or 0
        except Exception:
            rds_count = 0

        if rds_count >= total:
            log.info("  %-42s  already complete (%s rows) — skipped",
                     table, f"{rds_count:,}")
            return 0
        if rds_count > 0:
            log.info("  %-42s  resuming from row %s / %s",
                     table, f"{rds_count:,}", f"{total:,}")

    # Truncate (unless resuming mid-table)
    if not resume or rds_count == 0:
        with engine.begin() as c:
            c.execute(text(f"TRUNCATE TABLE {PG_SCHEMA}.{table} CASCADE"))

    # Column list — skip 'id' on core tables
    all_cols = duck.execute(
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_name='{table}' ORDER BY ordinal_position"
    ).df()["column_name"].tolist()

    insert_cols = [
        col for col in all_cols
        if not (col == "id" and table in CORE_TABLES)
    ]
    cols_sql   = ", ".join(f'"{c}"' for c in insert_cols)
    select_sql = ", ".join(f'"{c}"' for c in insert_cols)

    n_chunks = math.ceil((total - rds_count) / CHUNK)
    loaded   = 0

    for i in tqdm(range(n_chunks),
                  desc=f"  {table[:40]:<40}",
                  unit="chunk",
                  leave=True):

        offset = rds_count + i * CHUNK
        df = duck.execute(
            f"SELECT {select_sql} FROM {table} "
            f"LIMIT {CHUNK} OFFSET {offset}"
        ).df()

        if df.empty:
            break

        df     = clean_df(df)
        values = [tuple(r) for r in df.itertuples(index=False, name=None)]
        insert_chunk(engine, table, cols_sql, values)
        loaded += len(values)

    elapsed = time.time() - t_start
    rate    = loaded / elapsed if elapsed > 0 else 0
    log.info("  ✓ %-42s  %9s rows  %6.0f rows/s  %.0fs",
             table, f"{loaded:,}", rate, elapsed)
    return loaded


# ════════════════════════════════════════════════════════════════
#  VERIFICATION
# ════════════════════════════════════════════════════════════════

def verify(duck, engine, tables):
    log.info("")
    log.info("═" * 74)
    log.info("  VERIFICATION  — DuckDB vs RDS")
    log.info("  %-40s  %12s  %12s  %s",
             "Table", "DuckDB", "RDS", "Status")
    log.info("  " + "─" * 70)
    all_ok = True
    with engine.connect() as pg:
        for t in tables:
            try:
                duck_n = duck.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                continue
            try:
                rds_n = pg.execute(
                    text(f"SELECT COUNT(*) FROM {PG_SCHEMA}.{t}")
                ).scalar()
            except Exception:
                rds_n = "ERR"
            ok = "✓" if duck_n == rds_n else "✗ MISMATCH"
            if duck_n != rds_n:
                all_ok = False
            log.info("  %-40s  %12s  %12s  %s",
                     t,
                     f"{duck_n:,}",
                     f"{rds_n:,}" if isinstance(rds_n, int) else str(rds_n),
                     ok)
    log.info("═" * 74)
    if all_ok:
        log.info("  All tables match ✓")
    else:
        log.warning("  Mismatches — re-run with --resume to fill gaps")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stream waze_cargo.duckdb → AWS RDS PostgreSQL"
    )
    ap.add_argument(
        "--only",
        choices=["all", "lookups", "core", "ml"],
        default="all",
        help="Which group to migrate (default: all)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip fully-loaded tables; continue mid-table from last row",
    )
    args = ap.parse_args()

    duck   = open_duckdb()
    engine = open_rds()

    groups = []
    if args.only in ("all", "lookups"): groups.append(("LOOKUPS", LOOKUP_TABLES))
    if args.only in ("all", "core"):    groups.append(("CORE",    CORE_TABLES))
    if args.only in ("all", "ml"):      groups.append(("ML",      ML_TABLES))

    all_tables  = []
    grand_total = 0

    for group_name, tables in groups:
        log.info("")
        log.info("── %s (%d tables) %s",
                 group_name, len(tables), "─" * max(0, 55 - len(group_name)))
        for t in tables:
            ensure_table(duck, engine, t)
            n = migrate_table(duck, engine, t, resume=args.resume)
            grand_total += n
            all_tables.append(t)

    verify(duck, engine, all_tables)

    log.info("")
    log.info("Grand total migrated: %s rows", f"{grand_total:,}")
    duck.close()
    engine.dispose()
    log.info("Done.")


if __name__ == "__main__":
    main()
