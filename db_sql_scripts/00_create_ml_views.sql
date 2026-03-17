CREATE TABLE structured.clean_maritime_exports AS
SELECT
    e."PERIODO"::INTEGER                                    AS periodo,
    e."MES"::INTEGER                                        AS mes,
    e.year                                                  AS year,
    e."COD_ADUANA_TRAMITACION"                              AS cod_aduana,
    adu.nombre_aduana                                       AS aduana,
    e."COD_REGION_ORIGEN"                                   AS cod_region_origen,
    e."COD_PUERTO_EMBARQUE"                                 AS puerto_embarque,
    pu_emb.nombre_puerto                                    AS nombre_puerto_embarque,
    e."COD_PUERTO_DESEMBARQUE"                              AS puerto_desembarque,
    pu_des.nombre_puerto                                    AS nombre_puerto_desembarque,
    e."COD_PAIS_DESTINO"                                    AS cod_pais_destino,
    pais_dest.nombre_pais                                   AS pais_destino,
    pais_dest.nombre_continente                             AS continente_destino,
    e."COD_TIPO_OPERACION"                                  AS cod_tipo_operacion,
    e."COD_VIA_TRANSPORTE"                                  AS cod_via_transporte,
    e."COD_MODALIDAD_VENTA"                                 AS cod_modalidad_venta,
    e."CLAUSULA_VENTA"                                      AS clausula_venta,
    e."COD_TIPO_CARGA"                                      AS cod_tipo_carga,
    tc.nombre_tipo_carga                                    AS tipo_carga,
    e."ITEM_SA"                                             AS item_sa,
    SUBSTR(LPAD(e."ITEM_SA", 8, '0'), 1, 2)                 AS hs2_capitulo,
    SUBSTR(LPAD(e."ITEM_SA", 8, '0'), 1, 4)                 AS hs4_partida,
    SUBSTR(LPAD(e."ITEM_SA", 8, '0'), 1, 6)                 AS hs6_subpartida,
    hs.description                                          AS descripcion_producto,
    NULLIF(REPLACE(e."FOB_US_DUSLEG", ',', '.'), '')::NUMERIC       AS fob_us,
    NULLIF(REPLACE(e."FOBUS_AJUSTADO_IVV", ',', '.'), '')::NUMERIC  AS fob_us_ajustado,
    NULLIF(REPLACE(e."PESO_BRUTO_KG", ',', '.'), '')::NUMERIC       AS peso_bruto_kg,
    NULLIF(REPLACE(e."CANTIDAD_MERCANCIA", ',', '.'), '')::NUMERIC  AS cantidad_mercancia,
    e."COD_UNIDAD_MEDIDA"                                   AS cod_unidad_medida,
    e."MONEDA"                                              AS moneda
FROM structured.all_exports e
LEFT JOIN structured.lkp_aduanas adu
    ON e."COD_ADUANA_TRAMITACION" = adu.cod_aduana_tramitacion::TEXT
LEFT JOIN structured.lkp_puertos pu_emb
    ON e."COD_PUERTO_EMBARQUE" = pu_emb.cod_puerto::TEXT
LEFT JOIN structured.lkp_puertos pu_des
    ON e."COD_PUERTO_DESEMBARQUE" = pu_des.cod_puerto::TEXT
LEFT JOIN structured.lkp_paises pais_dest
    ON e."COD_PAIS_DESTINO" = pais_dest.cod_pais::TEXT
LEFT JOIN structured.lkp_tipos_carga tc
    ON e."COD_TIPO_CARGA" = tc.cod_tipo_carga::TEXT
LEFT JOIN structured.lkp_harmonized_system hs
    ON SUBSTR(LPAD(e."ITEM_SA", 8, '0'), 1, 6) = LPAD(hs.hscode::TEXT, 6, '0')
WHERE e."COD_VIA_TRANSPORTE" = '1';

CREATE INDEX idx_cme_year ON structured.clean_maritime_exports(year);
CREATE INDEX idx_cme_periodo_mes ON structured.clean_maritime_exports(periodo, mes);
CREATE INDEX idx_cme_puerto ON structured.clean_maritime_exports(puerto_embarque);

COMMENT ON TABLE structured.clean_maritime_exports IS 
'Clean maritime exports with lowercase columns, lookup JOINs for names/continents, derived HS codes, and numeric types. Filtered for maritime transport only.';


---------
---------

CREATE TABLE structured.clean_maritime_imports AS
SELECT
    SPLIT_PART(i."PERIODO", ',', 1)::INTEGER                AS periodo,
    SPLIT_PART(i."MES", ',', 1)::INTEGER                    AS mes,
    i.year                                                  AS year,
    i."COD_ADUANA_TRAMITACION"                              AS cod_aduana,
    adu.nombre_aduana                                       AS aduana,
    i."COD_PAIS_ORIGEN"                                     AS cod_pais_origen,
    pais_orig.nombre_pais                                   AS pais_origen,
    pais_orig.nombre_continente                             AS continente_origen,
    i."COD_PAIS_ADQUISICION"                                AS cod_pais_adquisicion,
    i."COD_PUERTO_EMBARQUE"                                 AS puerto_embarque,
    pu_emb.nombre_puerto                                    AS nombre_puerto_embarque,
    i."COD_PUERTO_DESEMBARQUE"                              AS puerto_desembarque,
    pu_des.nombre_puerto                                    AS nombre_puerto_desembarque,
    i."COD_TIPO_OPERACION"                                  AS cod_tipo_operacion,
    i."COD_REGIMEN_IMPORTACION"                             AS cod_regimen_importacion,
    i."COD_VIA_TRANSPORTE"                                  AS cod_via_transporte,
    i."CL_COMPRA"                                           AS clausula_compra,
    i."TPO_CARGA"                                           AS cod_tipo_carga,
    tc.nombre_tipo_carga                                    AS tipo_carga,
    i."ITEM_SA"                                             AS item_sa,
    SUBSTR(LPAD(i."ITEM_SA", 8, '0'), 1, 2)                 AS hs2_capitulo,
    SUBSTR(LPAD(i."ITEM_SA", 8, '0'), 1, 4)                 AS hs4_partida,
    SUBSTR(LPAD(i."ITEM_SA", 8, '0'), 1, 6)                 AS hs6_subpartida,
    hs.description                                          AS descripcion_producto,
    NULLIF(REPLACE(i."CIF_US", ',', '.'), '')::NUMERIC              AS cif_us,
    NULLIF(REPLACE(i."AD_VALOREM_US", ',', '.'), '')::NUMERIC       AS ad_valorem_us,
    NULLIF(REPLACE(i."CANTIDAD_MERCANCIA", ',', '.'), '')::NUMERIC  AS cantidad_mercancia,
    i."COD_UNIDAD_MEDIDA"                                   AS cod_unidad_medida,
    i."MONEDA"                                              AS moneda
FROM structured.all_imports i
LEFT JOIN structured.lkp_aduanas adu 
    ON i."COD_ADUANA_TRAMITACION" = adu.cod_aduana_tramitacion::TEXT
LEFT JOIN structured.lkp_paises pais_orig 
    ON i."COD_PAIS_ORIGEN" = pais_orig.cod_pais::TEXT
LEFT JOIN structured.lkp_puertos pu_emb 
    ON i."COD_PUERTO_EMBARQUE" = pu_emb.cod_puerto::TEXT
LEFT JOIN structured.lkp_puertos pu_des 
    ON i."COD_PUERTO_DESEMBARQUE" = pu_des.cod_puerto::TEXT
LEFT JOIN structured.lkp_tipos_carga tc 
    ON i."TPO_CARGA" = tc.cod_tipo_carga::TEXT
LEFT JOIN structured.lkp_harmonized_system hs 
    ON SUBSTR(LPAD(i."ITEM_SA", 8, '0'), 1, 6) = LPAD(hs.hscode::TEXT, 6, '0')
WHERE i."COD_VIA_TRANSPORTE" = '1';

COMMENT ON TABLE structured.clean_maritime_imports IS 
'Clean maritime imports with lowercase columns, lookup JOINs for names/continents, derived HS codes, and numeric types. Filtered for maritime transport only (cod_via_transporte = 1). Source: structured.all_imports + lookups.';

CREATE INDEX idx_cmi_year ON structured.clean_maritime_imports(year);
CREATE INDEX idx_cmi_periodo_mes ON structured.clean_maritime_imports(periodo, mes);
CREATE INDEX idx_cmi_puerto ON structured.clean_maritime_imports(puerto_desembarque);


