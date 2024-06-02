import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    df_red_RAMA = pd.read_csv(
        'https://datos.cdmx.gob.mx/dataset/72f6b09e-30c5-4b1e-9090-80717e6aedef/resource/ebc079e5-bd11-4830-b595-14292f753575/download/rama_2023_05.csv'
    )
    df_red_RAMA

    return df_red_RAMA


@test
def test_output(output, *args) -> None:
    assert output is not None, 'No se obtuvieron datos'