WITH source AS (
    SELECT * FROM {{ ref('stg_contaminacion_agua') }}
),
renamed AS (
    SELECT
        id,
        gridcode,
        valores,
        {{ transform_to_polygon("geometry") }} AS geometry
    FROM source
)
SELECT * FROM renamed