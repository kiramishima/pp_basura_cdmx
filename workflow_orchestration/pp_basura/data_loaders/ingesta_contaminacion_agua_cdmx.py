import pandas as pd
import geopandas as gpd
import requests, zipfile, io
from shapely import wkt

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def create_wkt_element(geom):
    return wkt.dumps(geom)

@data_loader
def load_data(*args, **kwargs):
    url = 'https://datos.cdmx.gob.mx/dataset/e82bdcc9-613e-498c-a518-a4f67978d5f0/resource/898506cf-cf25-4d1f-8b6a-192a6a8145be/download/contaminacin-de-agua-en-la-ciudad-de-mxico-.json'
    gdf = gpd.read_file(url, crs='EPSG:4326')
    gdf['geometry'] = gdf['geometry'].apply(create_wkt_element)
    df_contaminacion = pd.DataFrame(gdf)

    return df_contaminacion


@test
def test_output(output, *args) -> None:
    assert output is not None, 'No se obtuvieron datos'
