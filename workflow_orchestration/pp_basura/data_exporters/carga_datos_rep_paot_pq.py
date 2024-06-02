from pandas import DataFrame
from geopandas import GeoDataFrame
import pyarrow as pa
import pyarrow.parquet as pq
import os
from geopandas_postgis import PostGIS

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(df: DataFrame, *args, **kwargs):
    table = pa.Table.from_pandas(df)
    #Asignamos el nombre de la tabla PQ
    table_name = 'reportes_paot'
    # Asignamos una ruta
    root_path = f'/pp/cdmx/{table_name}'
    # Asignamos el tipo de FileSystem
    lfs = pa.fs.LocalFileSystem()

    print('RP', root_path)
    # Creamos nuestros archivos parquet
    pq.write_to_dataset(
        table,
        root_path=root_path,
        partition_cols=['delegacion'],
        filesystem=lfs
    )


