WITH source AS (
    SELECT * FROM {{ ref('stg_bronce_tiraderos_clandestinos') }}
),
renamed AS (
    SELECT
        {{ dbt_utils.generate_surrogate_key(['fid', 'id_sedema', 'id_delegacion']) }} AS id,
        {{ dbt.safe_cast("fid", api.Column.translate_type("integer")) }} AS fid,
        {{ dbt.safe_cast("id_sedema", api.Column.translate_type("integer")) }} AS id_sedema,
        {{ dbt.safe_cast("id_delegacion", api.Column.translate_type("integer")) }} AS id_delegacion,
        {{ dbt.safe_cast("delegacion", api.Column.translate_type("string")) }} AS delegacion,
        {{ dbt.safe_cast("calle", api.Column.translate_type("string")) }} AS calle,
        {{ dbt.safe_cast("colonia", api.Column.translate_type("string")) }} AS colonia,
        {{ create_point2("longitud", "latitud") }} AS geopoint
    FROM source
)
SELECT * FROM renamed