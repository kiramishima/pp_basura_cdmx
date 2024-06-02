from pandas import DataFrame
import pyarrow as pa
import pyarrow.parquet as pq
import os


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(df: DataFrame, *args, **kwargs):
    # Creamos la tabla desde el DataFrame
    table = pa.Table.from_pandas(df)
    #Asignamos el nombre de la tabla PQ
    table_name = 'rama'
    # Asignamos una ruta
    root_path = f'/pp/cdmx/{table_name}'
    # Asignamos el tipo de FileSystem
    lfs = pa.fs.LocalFileSystem()
    # Creamos nuestros archivos parquet
    pq.write_to_dataset(
        table,
        root_path=root_path,
        partition_cols=['fecha'], # particionamos por fecha
        filesystem=lfs,
        max_partitions=3100
    )


