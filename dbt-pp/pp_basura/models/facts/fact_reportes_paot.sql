WITH source AS (
    SELECT * FROM {{ ref('silver_reportes_paot') }}
),
renamed as (
    SELECT
        delegacion,
        COUNT(1) AS total
    FROM source
    GROUP BY delegacion
)
SELECT * FROM renamed