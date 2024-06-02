import pandas as pd
import geopandas as gpd
import requests, zipfile, io
import os
from geopandas_postgis import PostGIS

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    #zip_file_url = 'https://datos.cdmx.gob.mx/dataset/4a6576ab-92f0-4311-bfd3-7ddb650da8ca/resource/586237a7-2655-4a97-b043-c08cf3168830/download/denuncias_realizadas_ante_la_paot.zip'
    #filePath = '/DATASETS/denuncias_realizadas_ante_la_paot/denuncias_realizadas_ante_la_paot.shp'
    #if not os.path.isfile(filePath):
    #    r = requests.get(zip_file_url)
    #    z = zipfile.ZipFile(io.BytesIO(r.content))
    #    z.extractall("/DATASETS")
    filePath = 'https://datos.cdmx.gob.mx/dataset/4a6576ab-92f0-4311-bfd3-7ddb650da8ca/resource/514994d3-14d7-423a-9f95-d613e70cbfb1/download/denuncias-realizadas-ante-la-paot.csv'
    return pd.read_csv(filePath)


@test
def test_output(output, *args) -> None:
    assert output is not None, 'No se obtuvieron datos'
