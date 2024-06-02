import pandas as pd
import re

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    print(data.columns)
    # Fill Nombres Vacios de los rios
    data['NOMBRE'] = data['NOMBRE'].fillna(value='NA')
    data.columns = [col.lower() for col in data.columns]
    return data


@test
def test_output(output, *args) -> None:
    assert not ('NOMBRE' in output.columns), 'The output is undefined'