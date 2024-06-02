import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    # Converti a Titipo DateTime
    data['fecha'] = pd.to_datetime(data['fecha'])
    data['fecha'] = data['fecha'].dt.strftime("%Y-%m-%d")

    # Renombrar columnas
    data.columns = [col.lower() for col in data.columns]

    return data
