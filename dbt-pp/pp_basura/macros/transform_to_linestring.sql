{#
    Convierte el tipo de dato TEXT a LINESTRING
#}

{% macro transform_to_linestring(geometry) -%}
    cdmx_monitor.ST_GeomFromText({{ dbt.safe_cast("geometry", api.Column.translate_type("string")) }}, 4326)
{%- endmacro %}