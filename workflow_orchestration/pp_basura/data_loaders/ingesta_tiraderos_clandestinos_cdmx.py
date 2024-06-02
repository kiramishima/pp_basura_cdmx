import pandas as pd
import requests, zipfile, io

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    # TODO: Cambiar cuando se solucione el bug de geopandas
    #zip_file_url = 'https://datos.cdmx.gob.mx/dataset/c5a40bb7-241c-4bb0-8a59-6218526ba01c/resource/88468cea-5505-4d25-bb45-0f591d2793a9/download/tiraderos_clandestinos_al_cierre_de_2017.zip'
    #r = requests.get(zip_file_url)
    #z = zipfile.ZipFile(io.BytesIO(r.content))
    #z.extractall("/DATASETS")
    filePath = 'https://datos.cdmx.gob.mx/dataset/c5a40bb7-241c-4bb0-8a59-6218526ba01c/resource/7041d302-e17b-4137-94a7-383b6b713761/download/tiraderos-clandestinos-al-cierre-de-2017.csv'

    return pd.read_csv(filePath)


@test
def test_output(output, *args) -> None:
    assert output is not None, 'No se obtuvieron datos'
