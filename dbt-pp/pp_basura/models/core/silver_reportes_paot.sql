WITH source AS (
    SELECT * FROM {{ ref('stg_bronce_reportes_paot') }}
),
renamed AS (
    SELECT
        {{ dbt.safe_cast("id", api.Column.translate_type("integer")) }} AS id,
        {{ dbt.safe_cast("expediente", api.Column.translate_type("string")) }} AS expediente,
        {{ dbt.safe_cast("tipo_de_denuncia", api.Column.translate_type("string")) }} AS tipo_denuncia,
        {{ dbt.safe_cast("estatus", api.Column.translate_type("string")) }} AS status,
        {{ dbt.safe_cast("tema", api.Column.translate_type("string")) }} AS categoria,
        {{ dbt.safe_cast("delegacion", api.Column.translate_type("string")) }} AS delegacion,
        {{ dbt.safe_cast("colonia", api.Column.translate_type("string")) }} AS colonia,
        {{ dbt.safe_cast("cp", api.Column.translate_type("string")) }} AS cp,
        {{ dbt.safe_cast("denunciante", api.Column.translate_type("string")) }} AS denunciante,
        {{ dbt.safe_cast("medio_de_recepcion", api.Column.translate_type("string")) }} AS medio_recepcion,
        {{ create_point("lat", "_long") }},
        TO_DATE(fecha_de_recepcion, 'DD/MM/YYYY') AS fecha_recepcion,
        TO_DATE(fecha_de_admision, 'DD/MM/YYYY') AS fecha_admision
    FROM source
)
SELECT * FROM renamed