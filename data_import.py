import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

import os
import matplotlib.pyplot as plt
import numpy as np


# IMPORTANDO DATOS

# Define la URL del archivo ZIP que contiene los datos de gatos y perros.
_URL = 'https://drive.google.com/file/d/1zxeCpvFdKxUsODUUIZA_mPERcz2ZGf8D/view?usp=drivesdk'

# Evitar descargar si ya está
dataset_dir = os.path.join(os.path.expanduser("~"), ".keras", "datasets", "data") 
if not os.path.exists(dataset_dir):
    zip_dir = tf.keras.utils.get_file('data.zip', origin=_URL, extract=True)

# Descarga y extrae el archivo ZIP, guardando el directorio donde se encuentra.
zip_dir = tf.keras.utils.get_file('data.zip', origin=_URL, extract=True)


# Crea el directorio base para los datos de gatos y perros combinados.
# Se utiliza os.path.dirname() para obtener el directorio padre de zip_dir.
# Luego, se crea un nuevo directorio llamado 'cats_and_dogs_filtered' dentro del directorio padre.
base_dir = os.path.join(os.path.dirname(zip_dir), 'data', 'data')

# Crea el directorio de entrenamiento dentro del directorio base.
train_dir = os.path.join(base_dir, 'entrenamiento')

# Crea el directorio de validación dentro del directorio base.
validation_dir = os.path.join(base_dir, 'validacion')

# Crea subdirectorios para gatos y perros dentro del directorio de entrenamiento.
train_simmental_dir = os.path.join(train_dir, 'Simmental')
train_salers_dir = os.path.join(train_dir, 'Salers')
train_normanda_dir = os.path.join(train_dir, 'Normanda')
train_montbeliarde_dir = os.path.join(train_dir, 'Montbeliarde')
train_limousin_dir = os.path.join(train_dir, 'Limousin')
train_jersey_dir = os.path.join(train_dir, 'Jersey')
train_holstein_dir = os.path.join(train_dir, 'Holstein')
train_charolesa_dir = os.path.join(train_dir, 'Charolesa')
train_blonde_de_aquitania_dir = os.path.join(train_dir, 'Blonde_de_Aquitania')
train_abundancia_dir = os.path.join(train_dir, 'Abundancia')

# Crea subdirectorios para gatos y perros dentro del directorio de validación.
validation_simmental_dir = os.path.join(validation_dir, 'Simmental')
validation_salers_dir = os.path.join(validation_dir, 'Salers')
validation_normanda_dir = os.path.join(validation_dir, 'Normanda')
validation_montbeliarde_dir = os.path.join(validation_dir, 'Montbeliarde')
validation_limousin_dir = os.path.join(validation_dir, 'Limousin')
validation_jersey_dir = os.path.join(validation_dir, 'Jersey')
validation_holstein_dir = os.path.join(validation_dir, 'Holstein')
validation_charolesa_dir = os.path.join(validation_dir, 'Charolesa')
validation_blonde_de_aquitania_dir = os.path.join(validation_dir, 'Blonde_de_Aquitania')
validation_abundancia_dir = os.path.join(validation_dir, 'Abundancia')

num_simmental_tr = len(os.listdir(train_simmental_dir))
num_salers_tr = len(os.listdir(train_salers_dir))
num_normanda_tr = len(os.listdir(train_normanda_dir))
num_montbeliarde_tr = len(os.listdir(train_montbeliarde_dir))
num_limousin_tr = len(os.listdir(train_limousin_dir))
num_jersey_tr = len(os.listdir(train_jersey_dir))
num_holstein_tr = len(os.listdir(train_holstein_dir))
num_charolesa_tr = len(os.listdir(train_charolesa_dir))
num_blonde_de_aquitania_tr = len(os.listdir(train_blonde_de_aquitania_dir))
num_abundancia_tr = len(os.listdir(train_abundancia_dir))

num_simmental_val = len(os.listdir(validation_simmental_dir))
num_salers_val = len(os.listdir(validation_salers_dir))
num_normanda_val = len(os.listdir(validation_normanda_dir))
num_montbeliarde_val = len(os.listdir(validation_montbeliarde_dir))
num_limousin_val = len(os.listdir(validation_limousin_dir))
num_jersey_val = len(os.listdir(validation_jersey_dir))
num_holstein_val = len(os.listdir(validation_holstein_dir))
num_charolesa_val = len(os.listdir(validation_charolesa_dir))
num_blonde_de_aquitania_val = len(os.listdir(validation_blonde_de_aquitania_dir))
num_abundancia_val = len(os.listdir(validation_abundancia_dir))
total_train = num_simmental_tr + num_salers_tr + num_normanda_tr + num_montbeliarde_tr + num_limousin_tr + num_jersey_tr + num_holstein_tr + num_charolesa_tr + num_blonde_de_aquitania_tr + num_abundancia_tr
total_val = num_simmental_val + num_salers_val + num_normanda_val + num_montbeliarde_val + num_limousin_val + num_jersey_val + num_holstein_val + num_charolesa_val + num_blonde_de_aquitania_val + num_abundancia_val

print('total training Simmental images:', num_simmental_tr)
print('total training Salers images:', num_salers_tr)
print('total training Normanda images:', num_normanda_tr)
print('total training Montbeliarde images:', num_montbeliarde_tr)
print('total training Limousin images:', num_limousin_tr)
print('total training Jersey images:', num_jersey_tr)
print('total training Holstein images:', num_holstein_tr)
print('total training Charolesa images:', num_charolesa_tr)
print('total training Blonde de Aquitania images:', num_blonde_de_aquitania_tr)
print('total training Abundancia images:', num_abundancia_tr)
print('total validation Simmental images:', num_simmental_val)
print('total validation Salers images:', num_salers_val)
print('total validation Normanda images:', num_normanda_val)
print('total validation Montbeliarde images:', num_montbeliarde_val)
print('total validation Limousin images:', num_limousin_val)
print('total validation Jersey images:', num_jersey_val)
print('total validation Holstein images:', num_holstein_val)
print('total validation Charolesa images:', num_charolesa_val)
print('total validation Blonde de Aquitania images:', num_blonde_de_aquitania_val)
print('total validation Abundancia images:', num_abundancia_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


# PREPARANDO LOS DATOS

BATCH_SIZE = 100
IMG_SHAPE  = 150

# Crea un generador de imágenes para el conjunto de datos de entrenamiento.
# Rescale=1./255 normaliza los valores de píxeles de las imágenes al rango [0,1].
train_image_generator = ImageDataGenerator(rescale=1./255)

# Crea un generador de imágenes para el conjunto de datos de validación.
# Rescale=1./255 normaliza los valores de píxeles de las imágenes al rango [0,1].
validation_image_generator = ImageDataGenerator(rescale=1./255)

# Genera lotes de datos de entrenamiento a partir de imágenes en un directorio.
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='categorical')

# Genera lotes de datos de validación a partir de imágenes en un directorio.
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),
                                                              class_mode='categorical')



# VISUALIZACIÓN

# Obtiene un lote de imágenes de entrenamiento y las etiquetas correspondientes.
# El segundo valor devuelto, '_', se usa para descartar las etiquetas ya que no son necesarias aquí.
sample_training_images, _ = next(train_data_gen)

# Define una función para trazar un conjunto de imágenes.
def plotImages(images_arr):
    # Crea una figura con subtramas dispuestas en una sola fila.
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()  # Aplana el arreglo de subtramas para facilitar el acceso.
    # Itera sobre las imágenes y los ejes correspondientes.
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)  # Muestra la imagen en el eje actual.
        ax.axis('off')
    plt.tight_layout()  # Ajusta automáticamente la disposición para evitar superposiciones.
    plt.show(block=False)  # Muestra la figura con las imágenes.
    plt.pause(3)
    plt.close()

# Llama a la función plotImages para trazar las primeras 5 imágenes del lote de entrenamiento.
plotImages(sample_training_images[:5])


# Exportar variables necesarias
__all__ = [
    'train_data_gen',
    'val_data_gen',
    'total_train',
    'total_val',
    'BATCH_SIZE',
    'IMG_SHAPE',
    'train_dir',
    'validation_dir',
    'train_simmental_dir',
    'train_salers_dir',
    'train_normanda_dir',
    'train_montbeliarde_dir',
    'train_limousin_dir',
    'train_jersey_dir',
    'train_holstein_dir',
    'train_charolesa_dir',
    'train_blonde_de_aquitania_dir',
    'train_abundancia_dir',
    'validation_simmental_dir',
    'validation_salers_dir',
    'validation_normanda_dir',
    'validation_montbeliarde_dir',
    'validation_limousin_dir',
    'validation_jersey_dir',
    'validation_holstein_dir',
    'validation_charolesa_dir',
    'validation_blonde_de_aquitania_dir',
    'validation_abundancia_dir'
]



