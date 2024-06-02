WITH source_redman AS (
    SELECT * FROM {{ ref('stg_bronce_redma') }}
),
source_rama AS (
    SELECT * FROM {{ ref('stg_bronce_rama') }}
),
renamed AS (
    SELECT
        TO_DATE({{ dbt.safe_cast("source_rama.fecha", api.Column.translate_type("string")) }}, 'YYYY-MM-DD') AS fecha,
        /* REDMA */
        COALESCE({{ dbt.safe_cast("co", api.Column.translate_type("float")) }}, 0) AS redma_co,
        COALESCE({{ dbt.safe_cast("_no", api.Column.translate_type("float")) }}, 0) AS redma_no,
        COALESCE({{ dbt.safe_cast("no2", api.Column.translate_type("float")) }}, 0) AS redma_no2,
        COALESCE({{ dbt.safe_cast("nox", api.Column.translate_type("float")) }}, 0) AS redma_nox,
        COALESCE({{ dbt.safe_cast("o3", api.Column.translate_type("float")) }}, 0) AS redma_o3,
        COALESCE({{ dbt.safe_cast("source_redman.pm10", api.Column.translate_type("float")) }}, 0) AS redma_pm10,
        COALESCE({{ dbt.safe_cast("source_redman.pm25", api.Column.translate_type("float")) }}, 0) AS redma_pm25,
        COALESCE({{ dbt.safe_cast("so2", api.Column.translate_type("float")) }}, 0) AS redma_so2,
        /* RAMA */
        COALESCE({{ dbt.safe_cast("pbpst", api.Column.translate_type("float")) }}, 0) AS rama_pbpst,
        COALESCE({{ dbt.safe_cast("source_rama.pm10", api.Column.translate_type("float")) }}, 0) AS rama_pm10,
        COALESCE({{ dbt.safe_cast("source_rama.pm25", api.Column.translate_type("float")) }}, 0) AS rama_pm25,
        COALESCE({{ dbt.safe_cast("pst", api.Column.translate_type("float")) }}, 0) AS rama_pst
    FROM source_redman
    INNER JOIN source_rama ON source_rama.fecha = source_redman.fecha
)
SELECT * FROM renamed