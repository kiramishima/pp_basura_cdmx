import pandas as pd
import re

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    # Eliminamos los vacios
    data = data[data.longitud.notna()]
    print(data.info())
    # Seleccion de columnas
    cols = ['fid', 'id_sedema',	'no_alcaldi', 'alcaldia', 'calle', 'colonia', 'longitud', 'latitud', 'geo_point']
    data = data[cols]
    # Renombramos Alcaldia por delegación
    data.rename(columns={"alcaldia": "delegacion", "no_alcaldi": "id_delegacion"}, inplace=True)
    return data


@test
def test_output(output, *args) -> None:
    assert len(output.columns) ==  9, 'Hay más de 7 columnas definidas'