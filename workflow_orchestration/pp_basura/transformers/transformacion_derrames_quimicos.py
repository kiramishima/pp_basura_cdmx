import re
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def convert_snake(camel_input):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', camel_input)
    return '_'.join(map(str.lower, words))

@transformer
def transform(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    # print(data.shape)
    # Removemos la columna Total general
    data.drop(columns=['Total general'], inplace=True)
    # print(data.columns)
    # Eliminamos la fila CDMX
    data = data[data['Alcaldía'] != 'CDMX']
    # print('Delegaciones', data['Alcaldía'].unique())
    # Renombrar Alcaldía por delegacion
    data.rename(columns={"Alcaldía": "delegacion", "Derrame de químicos": "Derrame de quimicos",
    "Derrame o fuga de químicos": "Derrame o fuga de quimicos"}, inplace=True)
    # Renombramos nuestras columnas a snake_case. Ejemplo Derrame de diesel -> derrame_de_diesel
    data.columns = [convert_snake(col) for col in data.columns]
    # Specify your transformation logic here
    # print(data.info())
    return data


@test
def test_output(output, *args) -> None:
    assert not ('Total general' in output.columns), 'No se ha eliminado la variable "Total general"'
    assert not ('CDMX' in output['delegacion'].unique()), 'No se ha eliminado los datos para CDMX'
