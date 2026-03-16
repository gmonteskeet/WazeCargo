import json
import boto3
import psycopg2
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

BUCKET       = "wazecargo-263704545424-eu-north-1-an"
STAGING_BASE = "s3://wazecargo-263704545424-eu-north-1-an/staging/"
SECRET_NAME  = "wazecargo/postgresql"
REGION       = "eu-north-1"
SCHEMA       = "structured"

LOAD_MODE   = "full"
TARGET_YEAR = "2026"

sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session


def get_credentials():
    client = boto3.client("secretsmanager", region_name=REGION)
    secret = client.get_secret_value(SecretId=SECRET_NAME)
    data   = json.loads(secret["SecretString"])
    return {
        "host":     data["host"],
        "port":     int(data["port"]),
        "database": data["database"],
        "username": data["username"],
        "password": data["password"]
    }


def get_connection(creds):
    return psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        dbname=creds["database"],
        user=creds["username"],
        password=creds["password"],
        sslmode="require"
    )


def get_jdbc_url(creds):
    return (
        "jdbc:postgresql://" + creds["host"] + ":" + str(creds["port"]) +
        "/" + creds["database"] + "?sslmode=require"
    )


def get_jdbc_properties(creds):
    return {
        "user":     creds["username"],
        "password": creds["password"],
        "driver":   "org.postgresql.Driver"
    }


def setup_schema(creds):
    conn = get_connection(creds)
    conn.autocommit = True
    cur  = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS " + SCHEMA)
    print("Schema " + SCHEMA + " ready")
    cur.close()
    conn.close()


def create_index(creds, table_name):
    conn = get_connection(creds)
    conn.autocommit = True
    cur  = conn.cursor()
    index_name = "idx_" + table_name + "_year"
    cur.execute(
        "CREATE INDEX IF NOT EXISTS " + index_name +
        " ON " + SCHEMA + "." + table_name + "(year)"
    )
    print("Index created on " + SCHEMA + "." + table_name + ".year")
    cur.close()
    conn.close()


def check_year_in_table(creds, table_name, year):
    conn = get_connection(creds)
    conn.autocommit = True
    cur  = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM " + SCHEMA + "." + table_name + " WHERE year = %s",
        (int(year),)
    )
    count = cur.fetchone()[0]
    print("Rows currently in " + SCHEMA + "." + table_name + " for year=" + year + ": " + str(count))
    cur.close()
    conn.close()
    return count


def get_years_in_table(creds, table_name):
    conn = get_connection(creds)
    conn.autocommit = True
    cur  = conn.cursor()
    cur.execute(
        "SELECT year, COUNT(*) as row_count FROM " + SCHEMA + "." + table_name +
        " GROUP BY year ORDER BY year"
    )
    rows = cur.fetchall()
    print("Years currently in " + SCHEMA + "." + table_name + ":")
    for row in rows:
        print("  year=" + str(row[0]) + "  rows=" + str(row[1]))
    cur.close()
    conn.close()


def delete_year_from_table(creds, table_name, year):
    conn = get_connection(creds)
    conn.autocommit = True
    cur  = conn.cursor()
    cur.execute(
        "DELETE FROM " + SCHEMA + "." + table_name + " WHERE year = %s",
        (int(year),)
    )
    deleted = cur.rowcount
    print("Deleted " + str(deleted) + " rows for year=" + year + " from " + SCHEMA + "." + table_name)
    cur.close()
    conn.close()
    return deleted


def full_load(tipo, table_name, creds):
    path = STAGING_BASE + tipo + "/"
    print("FULL LOAD - Reading all staging Parquet from: " + path)

    df = spark.read.parquet(path)
    print("Columns found: " + str(df.columns))

    total = df.count()
    print("Total rows for " + tipo + ": " + str(total))

    if total == 0:
        print("No rows found for " + tipo + " - skipping")
        return

    df = df.withColumn("year", col("year").cast(IntegerType()))

    jdbc_url   = get_jdbc_url(creds)
    jdbc_props = get_jdbc_properties(creds)

    print("Writing to " + SCHEMA + "." + table_name + "...")
    df.repartition(2).write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", SCHEMA + "." + table_name) \
        .option("batchsize", 10000) \
        .mode("overwrite") \
        .options(**jdbc_props) \
        .save()

    print("Full load done - " + str(total) + " rows written to " + SCHEMA + "." + table_name)
    create_index(creds, table_name)
    get_years_in_table(creds, table_name)


def incremental_load(tipo, table_name, year, creds):
    print("INCREMENTAL LOAD - year=" + year + " for " + tipo)

    print("Before update:")
    check_year_in_table(creds, table_name, year)

    delete_year_from_table(creds, table_name, year)

    path = STAGING_BASE + tipo + "/year=" + year + "/"
    print("Reading staging Parquet from: " + path)

    df = spark.read.parquet(path)
    print("Columns found: " + str(df.columns))

    total = df.count()
    print("Total rows to insert for year=" + year + ": " + str(total))

    if total == 0:
        print("No rows found for year=" + year + " - skipping insert")
        return

    df = df.withColumn("year", col("year").cast(IntegerType()))

    jdbc_url   = get_jdbc_url(creds)
    jdbc_props = get_jdbc_properties(creds)

    print("Inserting into " + SCHEMA + "." + table_name + "...")
    df.repartition(2).write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", SCHEMA + "." + table_name) \
        .option("batchsize", 10000) \
        .mode("append") \
        .options(**jdbc_props) \
        .save()

    print("Incremental load done - " + str(total) + " rows inserted for year=" + year)

    print("After update:")
    check_year_in_table(creds, table_name, year)
    get_years_in_table(creds, table_name)


print("WazeCargo staging to RDS job starting")
print("Mode: " + LOAD_MODE)
if LOAD_MODE == "incremental":
    print("Target year: " + TARGET_YEAR)

creds = get_credentials()
print("Credentials retrieved from secret: " + SECRET_NAME)
print("Connecting to: " + creds["host"] + "/" + creds["database"])

setup_schema(creds)

if LOAD_MODE == "full":
    print("Running FULL LOAD for all years...")
    full_load("ingresos", "all_imports", creds)
    full_load("salidas", "all_exports", creds)

elif LOAD_MODE == "incremental":
    print("Running INCREMENTAL LOAD for year=" + TARGET_YEAR + "...")
    incremental_load("ingresos", "all_imports", TARGET_YEAR, creds)
    incremental_load("salidas", "all_exports", TARGET_YEAR, creds)

else:
    print("ERROR: LOAD_MODE must be full or incremental")

print("WazeCargo staging to RDS job complete")
print("Tables ready:")
print("  " + SCHEMA + ".all_imports")
print("  " + SCHEMA + ".all_exports")