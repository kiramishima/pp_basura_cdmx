import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    # Fuente
    url = 'https://datos.cdmx.gob.mx/dataset/3445b694-d426-4402-8942-54dd193b4dbc/resource/fb42a479-4241-4298-bb54-0b6bda72f88e/download/derrames_quimicos_2017_alcaldias_cdmx_81.csv'
    # Tipo de datos
    derrames_dtypes = {
        'Derrame de diesel': pd.Int64Dtype(),
        'Derrame de gasolina': pd.Int64Dtype(),
        'Derrame de mercaptano': pd.Int64Dtype(),
        'Derrame de químicos': pd.Int64Dtype(),
        'Derrame en ductos o poliductos': pd.Int64Dtype(),
        'Derrame o fuga de químicos': pd.Int64Dtype(),
        'Total general': pd.Int64Dtype()
    }

    return pd.read_csv(url, dtype=derrames_dtypes)


@test
def test_output(output, *args) -> None:
    assert output is not None, 'No se obtuvieron datos'
