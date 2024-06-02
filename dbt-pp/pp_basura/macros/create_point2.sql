{#
    Crea un GEOPOINT con latitud y longituf
#}

{% macro create_point2(lat, _long) -%}
    cdmx_monitor.ST_POINT({{ dbt.safe_cast(
        "longitud", api.Column.translate_type("float")) }},
        {{ dbt.safe_cast("latitud", api.Column.translate_type("float")) }}, 4326)
{%- endmacro %}