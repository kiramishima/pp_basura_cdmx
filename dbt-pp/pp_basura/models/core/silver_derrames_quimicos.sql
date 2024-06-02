WITH source AS (
    SELECT * FROM {{ ref('stg_bronce_derrames_quimicos') }}
),
renamed AS (
    SELECT
        {{ dbt_utils.generate_surrogate_key(['delegacion', 'derrame_de_diesel', 'derrame_de_gasolina', 'derrame_de_mercaptano', 'derrame_de_quimicos', 'derrame_en_ductos_o_poliductos', 'derrame_o_fuga_de_quimicos']) }} AS id,
        {{ dbt.safe_cast("delegacion", api.Column.translate_type("string")) }},
        COALESCE({{ dbt.safe_cast("derrame_de_diesel", api.Column.translate_type("integer")) }}, 0) AS num_derrame_diesel,
        COALESCE({{ dbt.safe_cast("derrame_de_gasolina", api.Column.translate_type("integer")) }}, 0) AS num_derrame_gasolina,
        COALESCE({{ dbt.safe_cast("derrame_de_mercaptano", api.Column.translate_type("integer")) }}, 0) AS num_derrame_mercaptano,
        COALESCE({{ dbt.safe_cast("derrame_de_quimicos", api.Column.translate_type("integer")) }}, 0) AS num_derrame_quimicos,
        COALESCE({{ dbt.safe_cast("derrame_en_ductos_o_poliductos", api.Column.translate_type("integer")) }}, 0) AS num_derrame_ductos_o_poliductos,
        COALESCE({{ dbt.safe_cast("derrame_o_fuga_de_quimicos", api.Column.translate_type("integer")) }}, 0) AS num_fuga_quimicos
    FROM source
)
SELECT * FROM renamed