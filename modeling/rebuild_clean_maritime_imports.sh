#!/usr/bin/env bash
# =============================================================================
# modeling/rebuild_clean_maritime_imports.sh
# =============================================================================
#
# Full reload of maritime.clean_maritime_imports from structured.all_imports
# + structured.lkp_*.
#
# Independent of the exports rebuild — runs separately.
#
# Year-by-year loop, each year in its own transaction to keep WAL bounded
# on db.t3.micro. Maritime-only, excludes airports.
#
# Run time: ~3 hours on db.t3.micro for the full 2002-2026 history.
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

# --- Truncate target ---------------------------------------------------------

log "Truncating maritime.clean_maritime_imports..."
PSQL -c "TRUNCATE TABLE maritime.clean_maritime_imports RESTART IDENTITY;"

# --- Discover years ----------------------------------------------------------

YEARS=$(PSQL -At -c "SELECT DISTINCT year FROM structured.all_imports ORDER BY year;")
log "Years to process: $(echo "$YEARS" | tr '\n' ' ')"

# --- Year-by-year load -------------------------------------------------------

for yr in $YEARS; do
    log "IMPORTS year=$yr ..."
    PSQL -c "
INSERT INTO maritime.clean_maritime_imports (
    periodo, mes, aduana, regimen_importacion, sigla_regimen,
    pais_origen, continente_origen, puerto_embarque, zona_geo_embarque,
    puerto_desembarque, tipo_carga, clausula_compra, sigla_clausula,
    item_sa, hs6_subpartida, hs4_partida, hs2_capitulo, descripcion_producto,
    cif_us, cantidad_mercancia, unidad_medida, sigla_unidad)
SELECT
    ai.\"PERIODO\"::smallint, ai.\"MES\"::smallint,
    la.nombre_aduana,
    lr.nombre_rgimen_importacion, lr.sigla_rgimen_importacion,
    lp.nombre_pais, lp.nombre_continente,
    pe.nombre_puerto, pe.zona_geografica,
    pd.nombre_puerto,
    ltc.nombre_tipo_carga,
    lc.nombre_clausula, lc.sigla_clausula,
    ai.\"ITEM_SA\", LEFT(ai.\"ITEM_SA\",6), LEFT(ai.\"ITEM_SA\",4), LEFT(ai.\"ITEM_SA\",2),
    hs.description,
    NULLIF(NULLIF(REPLACE(ai.\"CIF_US\", ',', '.'), ''), 'NaN')::double precision,
    NULLIF(NULLIF(REPLACE(ai.\"CANTIDAD_MERCANCIA\", ',', '.'), ''), 'NaN')::double precision,
    lu.nombre_unidad_medida, lu.unidad_medida
FROM structured.all_imports ai
LEFT JOIN structured.lkp_aduanas             la  ON NULLIF(SPLIT_PART(TRIM(ai.\"COD_ADUANA_TRAMITACION\"),  ',',1), '')::integer = la.cod_aduana_tramitacion
LEFT JOIN structured.lkp_regimen_importacion lr  ON NULLIF(SPLIT_PART(TRIM(ai.\"COD_REGIMEN_IMPORTACION\"), ',',1), '')::integer = lr.cod_rgimen_importacion
LEFT JOIN structured.lkp_paises              lp  ON NULLIF(SPLIT_PART(TRIM(ai.\"COD_PAIS_ORIGEN\"),         ',',1), '')::integer = lp.cod_pais
LEFT JOIN structured.lkp_puertos             pe  ON NULLIF(SPLIT_PART(TRIM(ai.\"COD_PUERTO_EMBARQUE\"),     ',',1), '')::integer = pe.cod_puerto
LEFT JOIN structured.lkp_puertos             pd  ON NULLIF(SPLIT_PART(TRIM(ai.\"COD_PUERTO_DESEMBARQUE\"),  ',',1), '')::integer = pd.cod_puerto
LEFT JOIN structured.lkp_tipos_carga         ltc ON ai.\"TPO_CARGA\"                                                             = ltc.cod_tipo_carga
LEFT JOIN structured.lkp_clausulas           lc  ON NULLIF(SPLIT_PART(TRIM(ai.\"CL_COMPRA\"),               ',',1), '')::integer = lc.cl_compra
LEFT JOIN structured.lkp_harmonized_system   hs  ON LEFT(ai.\"ITEM_SA\",6) = hs.hscode AND hs.level = '6'
LEFT JOIN structured.lkp_unidades_medida     lu  ON NULLIF(SPLIT_PART(TRIM(ai.\"COD_UNIDAD_MEDIDA\"),       ',',1), '')::integer = lu.cod_unidad_medida
WHERE ai.year = $yr
  AND SPLIT_PART(TRIM(ai.\"COD_VIA_TRANSPORTE\"), ',',1) = '1'
  AND (pd.tipo_puerto  IS NULL OR pd.tipo_puerto  <> 'Aeropuerto')
  AND (pe.tipo_puerto  IS NULL OR pe.tipo_puerto  <> 'Aeropuerto')
  AND (pd.nombre_puerto IS NULL OR pd.nombre_puerto NOT ILIKE 'AEROP%')
  AND (pe.nombre_puerto IS NULL OR pe.nombre_puerto NOT ILIKE 'AEROP%');"
done

# --- Maintenance + sanity checks ---------------------------------------------

log "VACUUM ANALYZE..."
PSQL -c "VACUUM ANALYZE maritime.clean_maritime_imports;"

log "Final count:"
PSQL -c "SELECT COUNT(*) AS rows FROM maritime.clean_maritime_imports;"

log "Airport leak check (should be 0):"
PSQL -c "SELECT COUNT(*) FROM maritime.clean_maritime_imports
         WHERE puerto_embarque ILIKE 'AEROP%' OR puerto_desembarque ILIKE 'AEROP%';"

log "Done."
