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
def load_data(*args, **kwargs) -> pd.DataFrame:
    zip_file_url = 'https://datos.cdmx.gob.mx/dataset/f77aba04-efe7-4010-aac0-766b30973bf7/resource/be4de719-8696-4c87-8c31-d936b2d0366c/download/rios_cdmx-2.zip'
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("/DATASETS")

    gdf = gpd.read_file('/DATASETS/rios_cdmx/Ri╠üos de CDMX.shp', crs='EPSG:4326')
    gdf['geometry'] = gdf['geometry'].apply(create_wkt_element)
    df_contaminacion = pd.DataFrame(gdf)

    return df_contaminacion


@test
def test_output(output, *args) -> None:
    assert output is not None, 'No se obtuvieron datos'
