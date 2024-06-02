import os
import pickle
import click
import mlflow
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Librerias para
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam, Adamax
from keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras import regularizers

# Ignorar Warnings
import warnings
warnings.filterwarnings("ignore")

# Carga los datos de los DF
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Toma los datos del entrenamiento, validacion y test y genera las imagenes
def create_gens (train_df, valid_df, test_df, batch_size):

    # parametros para las imagenes
    img_size = (224, 224)
    channels = 3 # pueden ser BGR o Grayscale
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    # creamos el tamaño del lote de pruebas
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    # escalamos la imagen y regresamos una nueva
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function= scalar, horizontal_flip= True)
    ts_gen = ImageDataGenerator(preprocessing_function= scalar)

    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= False, batch_size= test_batch_size)

    return train_gen, valid_gen, test_gen

def show_images(gen):
    '''
    This function take the data generator and show sample of the images
    '''

    # return classes , images to be displayed
    g_dict = gen.class_indices        # defines dictionary {'class': index}
    classes = list(g_dict.keys())     # defines list of dictionary's kays (classes), classes names : string
    images, labels = next(gen)        # get a batch size samples from the generator

    # calculate number of displayed samples
    length = len(labels)        # length of batch size
    sample = min(length, 25)    # check if sample less than 25 images

    plt.figure(figsize= (20, 20))

    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255       # scales data to range (0 - 255)
        plt.imshow(image)
        index = np.argmax(labels[i])  # get image index
        class_name = classes[index]   # get class of image
        plt.title(class_name, color= 'blue', fontsize= 12)
        plt.axis('off')
    plt.show()

def create_model(train_gen, valid_gen):
    # Estructura del modelo
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(list(train_gen.class_indices.keys())) # Numero de capas para Dense

    # Instanciamos ImageNet
    base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

    model = Sequential([
        base_model,
        BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        Dense(256, kernel_regularizer= regularizers.L2(l2= 0.016), activity_regularizer= regularizers.L1(0.006),
                    bias_regularizer= regularizers.L1(0.006), activation= 'relu'),
        Dropout(rate= 0.45, seed= 123),
        Dense(class_count, activation= 'softmax')
    ])

    model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

    print(model.summary())
    epochs = 5
    with mlflow.start_run() as run:
        mlflow.set_tag("Team", "Data Heroes")
        # creamos el modelo
        result = model.fit(
            x=train_gen,
            epochs=epochs,
            verbose=0,
            callbacks=[mlflow.keras.MlflowCallback(run)]
        )
        print(result)
        validation_acc = np.amax(result.history['val_acc']) 
        # print(':', validation_acc)
        mlflow.tensorflow.log_model(model)
        mlflow.set_tag("Best validation acc of epoch")
        mlflow.log_metric("loss", -validation_acc)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Directorio donde se almacenan los datos preprocesados"
)
def run_train(data_path: str):
    # Instancia de MLFlow
    mlflow.tensorflow.autolog()
    mlflow.set_tracking_uri("http://localhost:5000")
    # Asignamos el experimento
    mlflow.set_experiment("InvOperaciones") # Si no existe lo crea el experimento.
    # Cargamos los datos
    train_df = load_pickle(os.path.join(data_path, "train.pkl"))
    valid_df = load_pickle(os.path.join(data_path, "val.pkl"))
    test_df = load_pickle(os.path.join(data_path, "val.pkl"))
    batch_size = 40
    train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, batch_size)
    # mostramos las imagenes del entrenamiento
    # show_images(train_gen)
    # create_model(train_gen, valid_gen)
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(list(train_gen.class_indices.keys())) # Numero de capas para Dense

    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(list(train_gen.class_indices.keys())) # Numero de capas para Dense

    # Instanciamos ImageNet
    base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

    model = Sequential([
        base_model,
        BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        Dense(256, kernel_regularizer= regularizers.L2(l2= 0.016), activity_regularizer= regularizers.L1(0.006),
                    bias_regularizer= regularizers.L1(0.006), activation= 'relu'),
        Dropout(rate= 0.45, seed= 123),
        Dense(class_count, activation= 'softmax')
    ])

    model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

    print(model.summary())
    epochs = 5
    with mlflow.start_run() as run:
        mlflow.set_tag("Team", "Data Heroes")
        # creamos el modelo
        result = model.fit(
            x=train_gen,
            epochs=epochs,
            verbose=0,
            callbacks=[mlflow.keras.MlflowCallback(run)]
        )
        print(result)
        validation_acc = np.amax(result.history['val_acc']) 
        # print(':', validation_acc)
        mlflow.tensorflow.log_model(model)
        mlflow.set_tag("Best validation acc of epoch")
        mlflow.log_metric("loss", -validation_acc)


if __name__ == '__main__':
    run_train()