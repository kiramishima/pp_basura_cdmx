import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data: pd.DataFrame, *args, **kwargs):
    # filtramos los vacios
    data = data[data.lat.notna()]
    # Seleccion de columnas
    cols = ['id', 'expediente', 'tipo_de_denuncia',	'estatus',	'tema',	'colonia',	'cp', 'denunciante',  'medio_de_recepcion', 'fecha_de_recepcion', 'fecha_de_admision', 'alcaldia', 'geopoint', 'lat', 'long']
    data = data[cols]
    print(data.columns)
    # Renombrar Alcaldía por delegacion
    data.rename(columns={"alcaldia": "delegacion"}, inplace=True)
    # creamos nueva columna date
    #data['fch_d_r'] = pd.to_datetime(data['fch_d_r'])
    #data['fh_reporte'] = data['fch_d_r'].dt.date
    print(data.columns)

    return data
