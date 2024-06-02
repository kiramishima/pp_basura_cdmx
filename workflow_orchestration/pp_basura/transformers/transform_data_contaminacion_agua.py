import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data: pd.DataFrame, *args, **kwargs):
    # Renombrar columnas
    data.columns = [col.lower() for col in data.columns]

    return data


@test
def test_output(output, *args) -> None:
    assert not ('GRIDCODE' in output.columns), 'Campos en mayusculas'