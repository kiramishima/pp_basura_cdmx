WITH source AS (
    SELECT * FROM {{ ref('silver_tiraderos_clandestinos') }}
),
renamed as (
    SELECT
        delegacion,
        COUNT(1) AS total
    FROM source
    GROUP BY delegacion
)
SELECT * FROM renamed