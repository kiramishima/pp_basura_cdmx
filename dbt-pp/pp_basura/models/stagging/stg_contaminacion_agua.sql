{{ config(materialized='view') }}

SELECT * FROM {{ source('stagging', 'bronce_contaminacion_agua') }}
