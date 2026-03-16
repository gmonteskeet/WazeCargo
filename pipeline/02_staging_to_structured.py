"""
WazeCargo Glue Job: Staging to PostgreSQL
==========================================
Loads Parquet files from S3 staging layer into PostgreSQL structured schema.

Load Mode: Replace by year (DELETE + INSERT)
- Deletes all rows for the year being loaded
- Inserts new rows from staging
- Safe to run multiple times

Source: s3://wazecargo-263704545424-eu-north-1-an/staging/{tipo}/year={year}/
Target: PostgreSQL structured.ingresos, structured.salidas
"""

import sys
import json
import boto3
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import lit

# =============================================================================
# CONFIGURATION
# =============================================================================

BUCKET = "wazecargo-263704545424-eu-north-1-an"
STAGING_BASE = f"s3://{BUCKET}/staging"
SECRET_NAME = "wazecargo/postgresql"
REGION = "eu-north-1"
SCHEMA = "structured"

# Tables to load
TABLES = ["ingresos", "salidas"]

# =============================================================================
# INITIALIZE SPARK AND CLIENTS
# =============================================================================

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

s3 = boto3.client("s3", region_name=REGION)
secrets = boto3.client("secretsmanager", region_name=REGION)

# =============================================================================
# GET DATABASE CREDENTIALS
# =============================================================================

def get_db_credentials():
    """Retrieve PostgreSQL credentials from Secrets Manager."""
    print(f"Retrieving credentials from secret: {SECRET_NAME}")
    
    response = secrets.get_secret_value(SecretId=SECRET_NAME)
    secret = json.loads(response["SecretString"])
    
    return {
        "host": secret["host"],
        "port": secret["port"],
        "database": secret["database"],
        "username": secret["username"],
        "password": secret["password"]
    }


def get_jdbc_url(creds):
    """Build JDBC URL for PostgreSQL."""
    return f"jdbc:postgresql://{creds['host']}:{creds['port']}/{creds['database']}"


def get_jdbc_properties(creds):
    """Build JDBC connection properties."""
    return {
        "user": creds["username"],
        "password": creds["password"],
        "driver": "org.postgresql.Driver"
    }

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def create_schema_if_not_exists(creds):
    """Create the structured schema if it doesn't exist."""
    import psycopg2
    
    conn = psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        database=creds["database"],
        user=creds["username"],
        password=creds["password"]
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    print(f"Schema '{SCHEMA}' ensured")
    
    cursor.close()
    conn.close()


def delete_year_from_table(creds, table, year):
    """Delete all rows for a specific year from a table."""
    import psycopg2
    
    conn = psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        database=creds["database"],
        user=creds["username"],
        password=creds["password"]
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    full_table = f"{SCHEMA}.{table}"
    
    # Check if table exists
    cursor.execute(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = '{SCHEMA}' 
            AND table_name = '{table}'
        )
    """)
    table_exists = cursor.fetchone()[0]
    
    if table_exists:
        cursor.execute(f"DELETE FROM {full_table} WHERE year = %s", (year,))
        deleted = cursor.rowcount
        print(f"Deleted {deleted} rows from {full_table} for year={year}")
    else:
        print(f"Table {full_table} does not exist yet - will be created on insert")
    
    cursor.close()
    conn.close()


def create_index_if_not_exists(creds, table):
    """Create index on year column for fast filtering."""
    import psycopg2
    
    conn = psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        database=creds["database"],
        user=creds["username"],
        password=creds["password"]
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    index_name = f"idx_{table}_year"
    full_table = f"{SCHEMA}.{table}"
    
    cursor.execute(f"""
        SELECT EXISTS (
            SELECT FROM pg_indexes 
            WHERE schemaname = '{SCHEMA}' 
            AND indexname = '{index_name}'
        )
    """)
    index_exists = cursor.fetchone()[0]
    
    if not index_exists:
        cursor.execute(f"CREATE INDEX {index_name} ON {full_table}(year)")
        print(f"Created index {index_name} on {full_table}")
    
    cursor.close()
    conn.close()

# =============================================================================
# LIST AVAILABLE YEARS IN STAGING
# =============================================================================

def get_staging_years(tipo):
    """Get list of years available in staging for a given type."""
    prefix = f"staging/{tipo}/"
    
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    
    years = []
    if "CommonPrefixes" in response:
        for cp in response["CommonPrefixes"]:
            # Extract year from "staging/ingresos/year=2024/"
            folder = cp["Prefix"]
            if "year=" in folder:
                year = folder.split("year=")[1].rstrip("/")
                years.append(year)
    
    return sorted(years)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_year(creds, tipo, year):
    """Load a single year from staging to PostgreSQL."""
    staging_path = f"{STAGING_BASE}/{tipo}/year={year}/"
    full_table = f"{SCHEMA}.{tipo}"
    
    print(f"\n{'='*60}")
    print(f"Loading: {tipo} year={year}")
    print(f"Source: {staging_path}")
    print(f"Target: {full_table}")
    print("="*60)
    
    # Check if parquet files exist
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"staging/{tipo}/year={year}/")
    if "Contents" not in response:
        print(f"No files found in staging for {tipo}/year={year} - skipping")
        return 0
    
    parquet_files = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".parquet")]
    if not parquet_files:
        print(f"No parquet files found - skipping")
        return 0
    
    # Read parquet from staging
    df = spark.read.parquet(staging_path)
    
    row_count = df.count()
    print(f"Rows to load: {row_count}")
    
    if row_count == 0:
        print("No rows to load - skipping")
        return 0
    
    # Delete existing data for this year
    delete_year_from_table(creds, tipo, year)
    
    # Write to PostgreSQL
    jdbc_url = get_jdbc_url(creds)
    jdbc_props = get_jdbc_properties(creds)
    
    df.write.jdbc(
        url=jdbc_url,
        table=full_table,
        mode="append",  # Append because we already deleted the year
        properties=jdbc_props
    )
    
    print(f"Successfully loaded {row_count} rows to {full_table}")
    
    # Create index after first load
    create_index_if_not_exists(creds, tipo)
    
    return row_count


def load_table(creds, tipo):
    """Load all years for a table type."""
    years = get_staging_years(tipo)
    
    if not years:
        print(f"\nNo data found in staging for {tipo}")
        return
    
    print(f"\nFound years in staging/{tipo}/: {years}")
    
    total_rows = 0
    for year in years:
        rows = load_year(creds, tipo, year)
        total_rows += rows
    
    print(f"\nTotal rows loaded to {SCHEMA}.{tipo}: {total_rows}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("WazeCargo Glue Job: Staging to PostgreSQL")
    print("="*60)
    print(f"Source: {STAGING_BASE}")
    print(f"Target Schema: {SCHEMA}")
    print(f"Tables: {TABLES}")
    print(f"Mode: Replace by year (DELETE + INSERT)")
    
    # Get credentials
    creds = get_db_credentials()
    print(f"\nConnecting to: {creds['host']}")
    
    # Ensure schema exists
    create_schema_if_not_exists(creds)
    
    # Load each table
    for tipo in TABLES:
        load_table(creds, tipo)
    
    print("\n" + "="*60)
    print("WazeCargo Glue Job: Complete")
    print("="*60)


if __name__ == "__main__":
    main()
