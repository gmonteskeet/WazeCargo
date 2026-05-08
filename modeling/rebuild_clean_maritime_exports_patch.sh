#!/usr/bin/env bash
# =============================================================================
# modeling/rebuild_clean_maritime_exports_patch.sh
# =============================================================================
#
# Re-loads ONLY maritime.clean_maritime_exports with a defensive regex on
# all integer casts. Used to recover from the 'invalid input syntax for
# type integer' error caused by whitespace-only values in two columns:
#
#   - COD_UNIDAD_MEDIDA: ~7,000 rows in years 2002-2009
#   - CLAUSULA_VENTA:    ~28,000 rows in years 2010-2020
#
# The new cast pattern is:
#   NULLIF(REGEXP_REPLACE(col, '[^0-9].*$', ''), '')::integer
# which strips everything from the first non-digit onward, returning NULL
# for whitespace-only or letter-code values instead of erroring.
#
# This script does NOT touch maritime.clean_maritime_imports — those rows
# loaded successfully under the old logic and don't need re-running.
#
# Run time: ~30-60 minutes on db.t3.micro for the full 2002-2026 history.
# =============================================================================

set -euo pipefail

SECRET_ID='arn:aws:secretsmanager:eu-north-1:263704545424:secret:rds!db-a7a252bf-a8e7-4147-80c6-159d1f33846b-dlG7HK'
HOST='wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com'
DBNAME='waze_cargo'
USER='postgresmasterWZ'

CONN="host=${HOST} port=5432 dbname=${DBNAME} user=${USER} sslmode=require"

# --- Pre-flight --------------------------------------------------------------

for cmd in aws jq psql; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: '$cmd' not on PATH" >&2; exit 1; }
done

aws sts get-caller-identity >/dev/null 2>&1 \
    || { echo "ERROR: AWS CLI not configured" >&2; exit 1; }

export PGPASSWORD
PGPASSWORD=$(aws secretsmanager get-secret-value \
    --secret-id "$SECRET_ID" \
    --query SecretString --output text \
    | jq -rj '.password')

PSQL() { psql "$CONN" -v ON_ERROR_STOP=1 "$@"; }
log()  { printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*"; }

# --- Truncate only exports (imports stays put) -------------------------------

log "Truncating maritime.clean_maritime_exports (imports untouched)..."
PSQL -c "TRUNCATE TABLE maritime.clean_maritime_exports RESTART IDENTITY;"

# --- Discover years to process -----------------------------------------------

YEARS=$(PSQL -At -c "SELECT DISTINCT year FROM structured.all_exports ORDER BY year;")
log "Years to process: $(echo "$YEARS" | tr '\n' ' ')"

# --- EXPORTS - year-by-year, with regex-protected integer casts --------------

for yr in $YEARS; do
    log "EXPORTS year=$yr ..."
    PSQL -c "
INSERT INTO maritime.clean_maritime_exports (
    periodo, mes, aduana, region_origen,
    puerto_embarque, zona_geo_embarque,
    pais_destino, continente_destino, puerto_desembarque,
    tipo_carga, clausula_venta, sigla_clausula,
    item_sa, hs6_subpartida, hs4_partida, hs2_capitulo, descripcion_producto,
    fob_us, peso_bruto_kg, cantidad_mercancia, unidad_medida, sigla_unidad)
SELECT
    ae.\"PERIODO\"::smallint, ae.\"MES\"::smallint,
    la.nombre_aduana,
    lreg.nombre_region,
    pe.nombre_puerto, pe.zona_geografica,
    lp.nombre_pais, lp.nombre_continente,
    pd.nombre_puerto,
    ltc.nombre_tipo_carga,
    lc.nombre_clausula, lc.sigla_clausula,
    ae.\"ITEM_SA\", LEFT(ae.\"ITEM_SA\",6), LEFT(ae.\"ITEM_SA\",4), LEFT(ae.\"ITEM_SA\",2),
    hs.description,
    NULLIF(NULLIF(REPLACE(ae.\"FOB_US_DUSLEG\",     ',', '.'), ''), 'NaN')::double precision,
    NULLIF(NULLIF(REPLACE(ae.\"PESO_BRUTO_KG\",     ',', '.'), ''), 'NaN')::double precision,
    NULLIF(NULLIF(REPLACE(ae.\"CANTIDAD_MERCANCIA\", ',', '.'), ''), 'NaN')::double precision,
    lu.nombre_unidad_medida, lu.unidad_medida
FROM structured.all_exports ae
LEFT JOIN structured.lkp_aduanas             la   ON NULLIF(REGEXP_REPLACE(ae.\"COD_ADUANA_TRAMITACION\", '[^0-9].*\$', ''), '')::integer = la.cod_aduana_tramitacion
LEFT JOIN structured.lkp_regiones            lreg ON NULLIF(REGEXP_REPLACE(ae.\"COD_REGION_ORIGEN\",      '[^0-9].*\$', ''), '')::integer = lreg.cod_region_origen
LEFT JOIN structured.lkp_puertos             pe   ON NULLIF(REGEXP_REPLACE(ae.\"COD_PUERTO_EMBARQUE\",    '[^0-9].*\$', ''), '')::integer = pe.cod_puerto
LEFT JOIN structured.lkp_puertos             pd   ON NULLIF(REGEXP_REPLACE(ae.\"COD_PUERTO_DESEMBARQUE\", '[^0-9].*\$', ''), '')::integer = pd.cod_puerto
LEFT JOIN structured.lkp_paises              lp   ON NULLIF(REGEXP_REPLACE(ae.\"COD_PAIS_DESTINO\",       '[^0-9].*\$', ''), '')::integer = lp.cod_pais
LEFT JOIN structured.lkp_tipos_carga         ltc  ON ae.\"COD_TIPO_CARGA\"                                                              = ltc.cod_tipo_carga
LEFT JOIN structured.lkp_clausulas           lc   ON NULLIF(REGEXP_REPLACE(ae.\"CLAUSULA_VENTA\",         '[^0-9].*\$', ''), '')::integer = lc.cl_compra
LEFT JOIN structured.lkp_harmonized_system   hs   ON LEFT(ae.\"ITEM_SA\",6) = hs.hscode AND hs.level = '6'
LEFT JOIN structured.lkp_unidades_medida     lu   ON NULLIF(REGEXP_REPLACE(ae.\"COD_UNIDAD_MEDIDA\",      '[^0-9].*\$', ''), '')::integer = lu.cod_unidad_medida
WHERE ae.year = $yr
  AND SPLIT_PART(TRIM(ae.\"COD_VIA_TRANSPORTE\"), ',',1) = '1'
  AND (pe.tipo_puerto  IS NULL OR pe.tipo_puerto  <> 'Aeropuerto')
  AND (pd.tipo_puerto  IS NULL OR pd.tipo_puerto  <> 'Aeropuerto')
  AND (pe.nombre_puerto IS NULL OR pe.nombre_puerto NOT ILIKE 'AEROP%')
  AND (pd.nombre_puerto IS NULL OR pd.nombre_puerto NOT ILIKE 'AEROP%');"
done

# --- Maintenance + sanity checks ---------------------------------------------

log "VACUUM ANALYZE..."
PSQL -c "VACUUM ANALYZE maritime.clean_maritime_exports;"

log "Final counts:"
PSQL -c "SELECT 'clean_maritime_imports' AS tbl, COUNT(*) AS rows FROM maritime.clean_maritime_imports
         UNION ALL
         SELECT 'clean_maritime_exports',         COUNT(*)        FROM maritime.clean_maritime_exports;"

log "Airport leak check (should be 0):"
PSQL -c "SELECT 'exports_aerop' AS chk, COUNT(*) FROM maritime.clean_maritime_exports
           WHERE puerto_embarque ILIKE 'AEROP%' OR puerto_desembarque ILIKE 'AEROP%';"

log "Lookup miss check (rows where the regex fix produced NULL on previously-bad columns):"
PSQL -c "SELECT
            COUNT(*) FILTER (WHERE periodo BETWEEN 2002 AND 2009 AND unidad_medida IS NULL) AS unidad_nulls_2002_2009,
            COUNT(*) FILTER (WHERE periodo BETWEEN 2010 AND 2020 AND clausula_venta IS NULL) AS clausula_nulls_2010_2020
         FROM maritime.clean_maritime_exports;"

log "Done."
