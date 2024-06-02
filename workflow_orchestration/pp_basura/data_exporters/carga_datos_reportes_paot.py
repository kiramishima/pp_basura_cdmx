from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.postgres import Postgres
from pandas import DataFrame
from os import path

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_postgres(df: DataFrame, **kwargs) -> None:
    schema = kwargs['schema_name']
    schema_name = schema 
    table_name = 'bronce_reportes_paot'
    config_path = path.join(get_repo_path(), 'io_config.yaml')
    config_profile = 'default'
    config = kwargs['db_uri']
    print('config', config)
    # df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x))
    print(df.info())
    # engine = create_engine(config, echo=False)
    #connection = engine.raw_connection()
    #df.to_sql(
    #    table_name, 
    #    connection,
    #    schema=schema_name,
    #    if_exists='replace',
    #    index=False
    #)
    
    with Postgres.with_config(ConfigFileLoader(config_path, config_profile)) as loader:
        loader.export(
            df,
            schema_name,
            table_name,
            index=False,  # Specifies whether to include index in exported table
            if_exists='replace',  # Specify resolution policy if table name already exists
        )
    # gdf.to_postgis("my_table", engine)  
    #with Postgres.with_config(ConfigFileLoader(config_path, config_profile)) as loader:
    #    loader.export(
    #        df,
    #        schema_name,
    #        table_name,
    #        index=False,  # Specifies whether to include index in exported table
    #        if_exists='replace',  # Specify resolution policy if table name already exists
    #        encoding='latin1'
    #    )