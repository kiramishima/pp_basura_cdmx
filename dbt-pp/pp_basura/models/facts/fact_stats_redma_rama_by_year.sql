WITH source AS (
    SELECT * FROM {{ ref('silver_rama_redma') }}
),
renamed as (
    SELECT
        DATE_PART('YEAR', fecha) AS year,
        /* RAMA */
        COALESCE(AVG(rama_pbpst), 0) AS avg_rama_pbpst,
        COALESCE(MIN(rama_pbpst), 0) AS min_rama_pbpst,
        COALESCE(MAX(rama_pbpst), 0) AS max_rama_pbpst,
        COALESCE(AVG(rama_pm10), 0) AS avg_rama_pm10,
        COALESCE(MIN(rama_pm10), 0) AS min_rama_pm10,
        COALESCE(MAX(rama_pm10), 0) AS max_rama_pm10,
        COALESCE(AVG(rama_pm25), 0) AS avg_rama_pm25,
        COALESCE(MIN(rama_pm25), 0) AS min_rama_pm25,
        COALESCE(MAX(rama_pm25), 0) AS max_rama_pm25,
        COALESCE(AVG(rama_pst), 0) AS avg_rama_pst,
        COALESCE(MIN(rama_pst), 0) AS min_rama_pst,
        COALESCE(MAX(rama_pst), 0) AS max_rama_pst,
        /* REDMA */
        COALESCE(AVG(redma_co), 0) AS avg_redma_co,
        COALESCE(MIN(redma_co), 0) AS min_redma_co,
        COALESCE(MAX(redma_co), 0) AS max_redma_co,
        COALESCE(AVG(redma_no), 0) AS avg_redma_no,
        COALESCE(MIN(redma_no), 0) AS min_redma_no,
        COALESCE(MAX(redma_no), 0) AS max_redma_no,
        COALESCE(AVG(redma_no2), 0) AS avg_redma_no2,
        COALESCE(MIN(redma_no2), 0) AS min_redma_no2,
        COALESCE(MAX(redma_no2), 0) AS max_redma_no2,
        COALESCE(AVG(redma_nox), 0) AS avg_redma_nox,
        COALESCE(MIN(redma_nox), 0) AS min_redma_nox,
        COALESCE(MAX(redma_nox), 0) AS max_redma_nox,
        COALESCE(AVG(redma_o3), 0) AS avg_redma_o3,
        COALESCE(MIN(redma_o3), 0) AS min_redma_o3,
        COALESCE(MAX(redma_o3), 0) AS max_redma_o3,
        COALESCE(AVG(redma_pm10), 0) AS avg_redma_pm10,
        COALESCE(MIN(redma_pm10), 0) AS min_redma_pm10,
        COALESCE(MAX(redma_pm10), 0) AS max_redma_pm10,
        COALESCE(AVG(redma_pm25), 0) AS avg_redma_pm25,
        COALESCE(MIN(redma_pm25), 0) AS min_redma_pm25,
        COALESCE(MAX(redma_pm25), 0) AS max_redma_pm25,
        COALESCE(AVG(redma_so2), 0) AS avg_redma_so2,
        COALESCE(MIN(redma_so2), 0) AS min_redma_so2,
        COALESCE(MAX(redma_so2), 0) AS max_redma_so2
    FROM source
    GROUP BY year
    ORDER BY year DESC
)
SELECT * FROM renamed