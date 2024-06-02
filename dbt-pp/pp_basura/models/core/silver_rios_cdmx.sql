WITH source AS (
    SELECT * FROM {{ ref('stg_bronce_rios_cdmx') }}
),
renamed AS (
    SELECT
        {{ dbt.safe_cast("id", api.Column.translate_type("integer")) }} AS id,
        {{ dbt.safe_cast("nombre", api.Column.translate_type("string")) }} AS nombre,
        {{ dbt.safe_cast("longitud", api.Column.translate_type("float")) }} AS longitud,
        {{ dbt.safe_cast("tipo", api.Column.translate_type("string")) }} AS tipo,
        {{ transform_to_linestring("geometry") }} AS geometry
    FROM source
)
SELECT * FROM renamed
  