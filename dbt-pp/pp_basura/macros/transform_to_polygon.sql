{#
    Convierte el tipo de dato TEXT a POLYGON
#}

{% macro transform_to_polygon(geometry) -%}
    cdmx_monitor.ST_GeomFromText({{ dbt.safe_cast("geometry", api.Column.translate_type("string")) }}, 4326)
{%- endmacro %}