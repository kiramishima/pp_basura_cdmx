import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    df_redma = pd.read_csv('https://datos.cdmx.gob.mx/dataset/1729ac65-a3c5-46ab-a851-71a983795598/resource/12769c73-8bb9-4244-ad60-2dc8e2ab7ddc/download/redma_2023_05.csv')

    return df_redma


@test
def test_output(output, *args) -> None:
    assert output is not None, 'No se obtuvieron datos'