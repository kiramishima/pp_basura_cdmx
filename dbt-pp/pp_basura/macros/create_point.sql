{#
    Crea un GEOPOINT con latitud y longituf
#}

{% macro create_point(lat, _long) -%}
    cdmx_monitor.ST_POINT({{ dbt.safe_cast(
        "_long", api.Column.translate_type("float")) }},
        {{ dbt.safe_cast("lat", api.Column.translate_type("float")) }}, 4326)
{%- endmacro %}