#!/usr/bin/env bash
# Rebuild waze_cargo.clean_maritime_imports / clean_maritime_exports
# Chunked by year: each year = one psql call = one transaction.
# Keeps WAL / temp bounded on db.t3.micro (99 GB allocated).

set -euo pipefail

SECRET_ID='arn:aws:secretsmanager:eu-north-1:263704545424:secret:rds!db-a7a252bf-a8e7-4147-80c6-159d1f33846b-dlG7HK'
CERT=/home/koko/Waze_Cargo/global-bundle.pem
CONN="host=wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com port=5432 dbname=waze_cargo user=postgresmasterWZ sslmode=verify-full sslrootcert=${CERT}"

export PGPASSWORD
PGPASSWORD=$(aws secretsmanager get-secret-value --secret-id "$SECRET_ID" --query SecretString --output text | jq -rj '.password')

PSQL() { psql "$CONN" -v ON_ERROR_STOP=1 "$@"; }

log() { printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*"; }

log "Truncating target tables..."
PSQL -c "TRUNCATE TABLE waze_cargo.clean_maritime_imports RESTART IDENTITY;
         TRUNCATE TABLE waze_cargo.clean_maritime_exports RESTART IDENTITY;"

YEARS=$(PSQL -At -c "SELECT DISTINCT year FROM structured.all_imports ORDER BY year;")

for yr in $YEARS; do
    log "IMPORTS year=$yr ..."
    PSQL -c "
INSERT INTO waze_cargo.clean_maritime_imports (
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

for yr in $YEARS; do
    log "EXPORTS year=$yr ..."
    PSQL -c "
INSERT INTO waze_cargo.clean_maritime_exports (
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
LEFT JOIN structured.lkp_aduanas             la   ON NULLIF(SPLIT_PART(TRIM(ae.\"COD_ADUANA_TRAMITACION\"), ',',1), '')::integer = la.cod_aduana_tramitacion
LEFT JOIN structured.lkp_regiones            lreg ON NULLIF(SPLIT_PART(TRIM(ae.\"COD_REGION_ORIGEN\"),      ',',1), '')::integer = lreg.cod_region_origen
LEFT JOIN structured.lkp_puertos             pe   ON NULLIF(SPLIT_PART(TRIM(ae.\"COD_PUERTO_EMBARQUE\"),    ',',1), '')::integer = pe.cod_puerto
LEFT JOIN structured.lkp_puertos             pd   ON NULLIF(SPLIT_PART(TRIM(ae.\"COD_PUERTO_DESEMBARQUE\"), ',',1), '')::integer = pd.cod_puerto
LEFT JOIN structured.lkp_paises              lp   ON NULLIF(SPLIT_PART(TRIM(ae.\"COD_PAIS_DESTINO\"),       ',',1), '')::integer = lp.cod_pais
LEFT JOIN structured.lkp_tipos_carga         ltc  ON ae.\"COD_TIPO_CARGA\"                                                       = ltc.cod_tipo_carga
LEFT JOIN structured.lkp_clausulas           lc   ON NULLIF(SPLIT_PART(TRIM(ae.\"CLAUSULA_VENTA\"),         ',',1), '')::integer = lc.cl_compra
LEFT JOIN structured.lkp_harmonized_system   hs   ON LEFT(ae.\"ITEM_SA\",6) = hs.hscode AND hs.level = '6'
LEFT JOIN structured.lkp_unidades_medida     lu   ON NULLIF(SPLIT_PART(TRIM(ae.\"COD_UNIDAD_MEDIDA\"),      ',',1), '')::integer = lu.cod_unidad_medida
WHERE ae.year = $yr
  AND SPLIT_PART(TRIM(ae.\"COD_VIA_TRANSPORTE\"), ',',1) = '1'
  AND (pe.tipo_puerto  IS NULL OR pe.tipo_puerto  <> 'Aeropuerto')
  AND (pd.tipo_puerto  IS NULL OR pd.tipo_puerto  <> 'Aeropuerto')
  AND (pe.nombre_puerto IS NULL OR pe.nombre_puerto NOT ILIKE 'AEROP%')
  AND (pd.nombre_puerto IS NULL OR pd.nombre_puerto NOT ILIKE 'AEROP%');"
done

log "VACUUM ANALYZE..."
PSQL -c "VACUUM ANALYZE waze_cargo.clean_maritime_imports;"
PSQL -c "VACUUM ANALYZE waze_cargo.clean_maritime_exports;"

log "Final counts:"
PSQL -c "SELECT 'clean_maritime_imports' AS tbl, COUNT(*) FROM waze_cargo.clean_maritime_imports
         UNION ALL
         SELECT 'clean_maritime_exports',         COUNT(*) FROM waze_cargo.clean_maritime_exports;
         SELECT 'imports_aerop' AS chk, COUNT(*) FROM waze_cargo.clean_maritime_imports
           WHERE puerto_embarque ILIKE 'AEROP%' OR puerto_desembarque ILIKE 'AEROP%'
         UNION ALL
         SELECT 'exports_aerop',        COUNT(*) FROM waze_cargo.clean_maritime_exports
           WHERE puerto_embarque ILIKE 'AEROP%' OR puerto_desembarque ILIKE 'AEROP%';"

log "Done."
