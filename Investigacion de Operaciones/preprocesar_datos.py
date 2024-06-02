import os
import pickle
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Almacena un modelo de ML en un archivo
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

# Genera las rutas de los folders de las imagenes con sus etiquetas
def create_paths_labels(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    return filepaths, labels

# Une las rutas generadas con etiquetas en un solo DataFrame
def define_df(files, classes):
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)

# Separa el DF en datos de entrenamiento, validacion y prueba
def split_data(data_dir):
    # training
    files, classes = create_paths_labels(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123, stratify= strat)

    # valid and test
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 123, stratify= strat)

    return train_df, valid_df, test_df


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str):
    try:
        print("raw_data_path -> ", raw_data_path)
        print("dest_path -> ", dest_path)
        train_df, valid_df, test_df = split_data(raw_data_path)
        # print(train_df.info())
        # print(train_df.head())
        # Creamos el folder dest_path en caso de que no exista
        os.makedirs(dest_path, exist_ok=True)

        # Guardamos los datos
        dump_pickle(train_df, os.path.join(dest_path, "train.pkl"))
        dump_pickle(valid_df, os.path.join(dest_path, "val.pkl"))
        dump_pickle(dest_path, os.path.join(dest_path, "test.pkl"))
    except:
        print("Ocurrio un error")

if __name__ == '__main__':
    run_data_prep()