#!/usr/bin/env bash
# Rebuild maritime.clean_maritime_imports / clean_maritime_exports on EC2.
# Fully self-contained: downloads the RDS CA bundle, fetches the password
# from Secrets Manager, and runs the SQL inline.
#
# Prerequisites on the EC2 instance: psql, aws cli, jq, curl.
# Usage: bash 03_rebuild_clean_maritime.sh

set -euo pipefail

log() { printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*"; }

# --- RDS CA bundle (download once to /tmp) ---
CERT=/tmp/global-bundle.pem
if [ ! -f "$CERT" ]; then
    log "Downloading RDS CA bundle..."
    curl -sS -o "$CERT" https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
fi

# --- Connection ---
SECRET_ID='arn:aws:secretsmanager:eu-north-1:263704545424:secret:rds!db-a7a252bf-a8e7-4147-80c6-159d1f33846b-dlG7HK'
export PGPASSWORD
PGPASSWORD=$(aws secretsmanager get-secret-value \
    --secret-id "$SECRET_ID" \
    --region eu-north-1 \
    --query SecretString --output text | jq -rj '.password')

CONN="host=wazecargo-db.czioqa62i3cf.eu-north-1.rds.amazonaws.com \
port=5432 dbname=waze_cargo user=postgresmasterWZ \
sslmode=verify-full sslrootcert=${CERT}"

PSQL() { psql "$CONN" -v ON_ERROR_STOP=1 "$@"; }

# --- Execute ---
log "Starting rebuild of maritime.clean_maritime_* ..."

PSQL <<'SQL'
\timing on

-- Truncate targets
BEGIN;
TRUNCATE TABLE maritime.clean_maritime_imports RESTART IDENTITY;
TRUNCATE TABLE maritime.clean_maritime_exports RESTART IDENTITY;
COMMIT;

-- IMPORTS — year-by-year
DO $$
DECLARE
    yr int;
    rc bigint;
BEGIN
    FOR yr IN SELECT DISTINCT year FROM structured.all_imports ORDER BY year LOOP
        INSERT INTO maritime.clean_maritime_imports (
            periodo, mes, aduana,
            regimen_importacion, sigla_regimen,
            pais_origen, continente_origen,
            puerto_embarque, zona_geo_embarque,
            puerto_desembarque,
            tipo_carga,
            clausula_compra, sigla_clausula,
            item_sa, hs6_subpartida, hs4_partida, hs2_capitulo, descripcion_producto,
            cif_us, cantidad_mercancia,
            unidad_medida, sigla_unidad
        )
        SELECT
            ai."PERIODO"::smallint,
            ai."MES"::smallint,
            la.nombre_aduana,
            lr.nombre_rgimen_importacion,
            lr.sigla_rgimen_importacion,
            lp.nombre_pais,
            lp.nombre_continente,
            pe.nombre_puerto,
            pe.zona_geografica,
            pd.nombre_puerto,
            ltc.nombre_tipo_carga,
            lc.nombre_clausula,
            lc.sigla_clausula,
            ai."ITEM_SA",
            LEFT(ai."ITEM_SA", 6),
            LEFT(ai."ITEM_SA", 4),
            LEFT(ai."ITEM_SA", 2),
            COALESCE(hs6.description, hs4_fb.description),
            NULLIF(NULLIF(REPLACE(ai."CIF_US", ',', '.'), ''), 'NaN')::double precision,
            NULLIF(NULLIF(REPLACE(ai."CANTIDAD_MERCANCIA", ',', '.'), ''), 'NaN')::double precision,
            lu.nombre_unidad_medida,
            lu.unidad_medida
        FROM structured.all_imports ai
        LEFT JOIN structured.lkp_aduanas             la  ON NULLIF(SPLIT_PART(TRIM(ai."COD_ADUANA_TRAMITACION"),  ',', 1), '')::integer = la.cod_aduana_tramitacion
        LEFT JOIN structured.lkp_regimen_importacion lr  ON NULLIF(SPLIT_PART(TRIM(ai."COD_REGIMEN_IMPORTACION"), ',', 1), '')::integer = lr.cod_rgimen_importacion
        LEFT JOIN structured.lkp_paises              lp  ON NULLIF(SPLIT_PART(TRIM(ai."COD_PAIS_ORIGEN"),         ',', 1), '')::integer = lp.cod_pais
        LEFT JOIN structured.lkp_puertos             pe  ON NULLIF(SPLIT_PART(TRIM(ai."COD_PUERTO_EMBARQUE"),     ',', 1), '')::integer = pe.cod_puerto
        LEFT JOIN structured.lkp_puertos             pd  ON NULLIF(SPLIT_PART(TRIM(ai."COD_PUERTO_DESEMBARQUE"),  ',', 1), '')::integer = pd.cod_puerto
        LEFT JOIN structured.lkp_tipos_carga         ltc ON ai."TPO_CARGA"                                                              = ltc.cod_tipo_carga
        LEFT JOIN structured.lkp_clausulas           lc  ON NULLIF(SPLIT_PART(TRIM(ai."CL_COMPRA"),               ',', 1), '')::integer = lc.cl_compra
        LEFT JOIN structured.lkp_harmonized_system   hs6 ON LEFT(ai."ITEM_SA", 6) = hs6.hscode AND hs6.level = '6'
        LEFT JOIN LATERAL (
            SELECT description FROM structured.lkp_harmonized_system
            WHERE LEFT(hscode, 4) = LEFT(ai."ITEM_SA", 4) AND level = '6'
            ORDER BY hscode LIMIT 1
        ) hs4_fb ON hs6.hscode IS NULL
        LEFT JOIN structured.lkp_unidades_medida     lu  ON NULLIF(SPLIT_PART(TRIM(ai."COD_UNIDAD_MEDIDA"),       ',', 1), '')::integer = lu.cod_unidad_medida
        WHERE ai.year = yr
          AND SPLIT_PART(TRIM(ai."COD_VIA_TRANSPORTE"), ',', 1) = '1'
          AND (pd.tipo_puerto  IS NULL OR pd.tipo_puerto  <> 'Aeropuerto')
          AND (pe.tipo_puerto  IS NULL OR pe.tipo_puerto  <> 'Aeropuerto')
          AND (pd.nombre_puerto IS NULL OR pd.nombre_puerto NOT ILIKE 'AEROP%')
          AND (pe.nombre_puerto IS NULL OR pe.nombre_puerto NOT ILIKE 'AEROP%');
        GET DIAGNOSTICS rc = ROW_COUNT;
        RAISE NOTICE 'imports year=% inserted=%', yr, rc;
    END LOOP;
END $$;

-- EXPORTS — year-by-year
DO $$
DECLARE
    yr int;
    rc bigint;
BEGIN
    FOR yr IN SELECT DISTINCT year FROM structured.all_exports ORDER BY year LOOP
        INSERT INTO maritime.clean_maritime_exports (
            periodo, mes, aduana,
            region_origen,
            puerto_embarque, zona_geo_embarque,
            pais_destino, continente_destino,
            puerto_desembarque,
            tipo_carga,
            clausula_venta, sigla_clausula,
            item_sa, hs6_subpartida, hs4_partida, hs2_capitulo, descripcion_producto,
            fob_us, peso_bruto_kg, cantidad_mercancia,
            unidad_medida, sigla_unidad
        )
        SELECT
            ae."PERIODO"::smallint,
            ae."MES"::smallint,
            la.nombre_aduana,
            lreg.nombre_region,
            pe.nombre_puerto,
            pe.zona_geografica,
            lp.nombre_pais,
            lp.nombre_continente,
            pd.nombre_puerto,
            ltc.nombre_tipo_carga,
            lc.nombre_clausula,
            lc.sigla_clausula,
            ae."ITEM_SA",
            LEFT(ae."ITEM_SA", 6),
            LEFT(ae."ITEM_SA", 4),
            LEFT(ae."ITEM_SA", 2),
            COALESCE(hs6.description, hs4_fb.description),
            NULLIF(NULLIF(REPLACE(ae."FOB_US_DUSLEG", ',', '.'),      ''), 'NaN')::double precision,
            NULLIF(NULLIF(REPLACE(ae."PESO_BRUTO_KG", ',', '.'),      ''), 'NaN')::double precision,
            NULLIF(NULLIF(REPLACE(ae."CANTIDAD_MERCANCIA", ',', '.'), ''), 'NaN')::double precision,
            lu.nombre_unidad_medida,
            lu.unidad_medida
        FROM structured.all_exports ae
        LEFT JOIN structured.lkp_aduanas             la   ON NULLIF(SPLIT_PART(TRIM(ae."COD_ADUANA_TRAMITACION"), ',', 1), '')::integer = la.cod_aduana_tramitacion
        LEFT JOIN structured.lkp_regiones            lreg ON NULLIF(SPLIT_PART(TRIM(ae."COD_REGION_ORIGEN"),      ',', 1), '')::integer = lreg.cod_region_origen
        LEFT JOIN structured.lkp_puertos             pe   ON NULLIF(SPLIT_PART(TRIM(ae."COD_PUERTO_EMBARQUE"),    ',', 1), '')::integer = pe.cod_puerto
        LEFT JOIN structured.lkp_puertos             pd   ON NULLIF(SPLIT_PART(TRIM(ae."COD_PUERTO_DESEMBARQUE"), ',', 1), '')::integer = pd.cod_puerto
        LEFT JOIN structured.lkp_paises              lp   ON NULLIF(SPLIT_PART(TRIM(ae."COD_PAIS_DESTINO"),       ',', 1), '')::integer = lp.cod_pais
        LEFT JOIN structured.lkp_tipos_carga         ltc  ON ae."COD_TIPO_CARGA"                                                        = ltc.cod_tipo_carga
        LEFT JOIN structured.lkp_clausulas           lc   ON NULLIF(SPLIT_PART(TRIM(ae."CLAUSULA_VENTA"),         ',', 1), '')::integer = lc.cl_compra
        LEFT JOIN structured.lkp_harmonized_system   hs6  ON LEFT(ae."ITEM_SA", 6) = hs6.hscode AND hs6.level = '6'
        LEFT JOIN LATERAL (
            SELECT description FROM structured.lkp_harmonized_system
            WHERE LEFT(hscode, 4) = LEFT(ae."ITEM_SA", 4) AND level = '6'
            ORDER BY hscode LIMIT 1
        ) hs4_fb ON hs6.hscode IS NULL
        LEFT JOIN structured.lkp_unidades_medida     lu   ON NULLIF(SPLIT_PART(TRIM(ae."COD_UNIDAD_MEDIDA"),      ',', 1), '')::integer = lu.cod_unidad_medida
        WHERE ae.year = yr
          AND SPLIT_PART(TRIM(ae."COD_VIA_TRANSPORTE"), ',', 1) = '1'
          AND (pe.tipo_puerto  IS NULL OR pe.tipo_puerto  <> 'Aeropuerto')
          AND (pd.tipo_puerto  IS NULL OR pd.tipo_puerto  <> 'Aeropuerto')
          AND (pe.nombre_puerto IS NULL OR pe.nombre_puerto NOT ILIKE 'AEROP%')
          AND (pd.nombre_puerto IS NULL OR pd.nombre_puerto NOT ILIKE 'AEROP%');
        GET DIAGNOSTICS rc = ROW_COUNT;
        RAISE NOTICE 'exports year=% inserted=%', yr, rc;
    END LOOP;
END $$;

VACUUM ANALYZE maritime.clean_maritime_imports;
VACUUM ANALYZE maritime.clean_maritime_exports;

SELECT 'clean_maritime_imports' AS tbl, COUNT(*) AS rows FROM maritime.clean_maritime_imports
UNION ALL
SELECT 'clean_maritime_exports',         COUNT(*)       FROM maritime.clean_maritime_exports;

SELECT 'imports_aerop' AS chk, COUNT(*) FROM maritime.clean_maritime_imports
  WHERE puerto_embarque ILIKE 'AEROP%' OR puerto_desembarque ILIKE 'AEROP%'
UNION ALL
SELECT 'exports_aerop',        COUNT(*) FROM maritime.clean_maritime_exports
  WHERE puerto_embarque ILIKE 'AEROP%' OR puerto_desembarque ILIKE 'AEROP%';
SQL

log "Done."
