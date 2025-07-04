import data_import
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
from data_import import (train_data_gen, val_data_gen, total_train, total_val, BATCH_SIZE, IMG_SHAPE, train_dir, validation_dir,
                         train_simmental_dir, train_salers_dir, train_normanda_dir, train_montbeliarde_dir, train_limousin_dir,
                         train_jersey_dir, train_holstein_dir, train_charolesa_dir, train_blonde_de_aquitania_dir, train_abundancia_dir,
                         validation_simmental_dir, validation_salers_dir, validation_normanda_dir, validation_montbeliarde_dir, validation_limousin_dir,
                         validation_jersey_dir, validation_holstein_dir, validation_charolesa_dir,validation_blonde_de_aquitania_dir, validation_abundancia_dir)

# CONSTRUCCIÓN DEL MODELO
import random

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)
tf.keras.utils.set_random_seed(123)

tf.config.experimental.enable_op_determinism()
tf.keras.backend.clear_session()

from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize
model = Sequential()

# Convolution
model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                      input_shape = (150, 150, 3), activation = 'relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size = (2,2)))

# 2nd Layer
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 3rd Layer
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 4th Layer
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 5th Layer
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# Flattening
model.add(Flatten())

# Full Connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(units = 10, activation = 'softmax'))  # 10 razas


# Compilar la CNN
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


model.summary()


# TRAINING

EPOCHS = 20

# Entrena el modelo utilizando el generador de datos de entrenamiento y validación.
# Se utiliza fit_generator en lugar de fit porque se están utilizando generadores de flujo de datos.
# train_data_gen es el generador de datos de entrenamiento.
# steps_per_epoch es el número total de pasos por época, calculado como total_train / BATCH_SIZE.
# epochs es el número de épocas de entrenamiento.
# validation_data es el generador de datos de validación.
# validation_steps es el número total de pasos de validación, calculado como total_val / BATCH_SIZE.

history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))))


# EVALUACIÓN

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.pause(5)
plt.close()


score = model.evaluate(val_data_gen, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Test Predictions
predictions = model.predict(val_data_gen, steps=int(np.ceil(total_val / float(BATCH_SIZE))))
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_data_gen.classes

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

confusion = confusion_matrix(true_classes, predicted_classes)
class_labels = list(val_data_gen.class_indices.keys())

print("Matriz de Confusión:")
print(confusion)

print("\nReporte de Clasificación:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# PREDICCIÓN

import matplotlib.image as mpimg

# Ruta al directorio que contiene las imágenes
#image_dir = "./imagenes"
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "imagenes")

# Obtener una lista de nombres de archivos de imagen en el directorio que terminan con '.jpg'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.jpeg')]

# Ordenar los nombres de los archivos de imagen alfabéticamente
image_files = sorted(image_files)

# Crear una figura con subplots para mostrar las imágenes
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()  # Aplana el arreglo de subtramas para facilitar el acceso

# Iterar sobre los nombres de los archivos de imagen y los ejes correspondientes
for img_file, ax in zip(image_files, axes):
    # Construir la ruta completa de la imagen
    img_path = os.path.join(image_dir, img_file)
    # Leer la imagen usando matplotlib
    img = mpimg.imread(img_path)
    # Mostrar la imagen en el eje correspondiente
    ax.imshow(img)
    # Establecer el título del eje como el nombre del archivo de imagen
    ax.set_title(img_file)

plt.tight_layout()
plt.pause(5)
plt.close()



# One by One
test_image_path = os.path.join(image_dir, 'vaca_1.jpg')
test_image = load_img(test_image_path, target_size=(150, 150))

test_image = img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

predicted_class = np.argmax(result)
class_labels = list(val_data_gen.class_indices.keys())
prediction = class_labels[predicted_class]


print(prediction)


# Definir la lista de nombres de las imágenes
image_names = ['vaca_1.jpg', 'vaca_2.jpeg', 'vaca_3.jpeg', 'vaca_4.jpeg', 'vaca_5.jpg', 'vaca_6.jpeg', 'vaca_7.jpg', 'vaca_8.jpeg']

# Crear una lista para almacenar las predicciones
predictions = []

# Iterar a través de las imágenes y hacer predicciones
for img_name in image_names:
    img_path = os.path.join(image_dir, img_name)
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)

    predicted_class = np.argmax(result)
    class_labels = list(val_data_gen.class_indices.keys())
    prediction = class_labels[predicted_class]

    predictions.append(prediction)

print(predictions)


# Crear una figura con subplots para mostrar las imágenes
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()

# Iterar sobre los nombres de los archivos de imagen, las etiquetas y los ejes correspondientes
for img_file, label, ax in zip(image_files, predictions, axes):
    # Construir la ruta completa de la imagen
    img_path = os.path.join(image_dir, img_file)
    # Leer la imagen usando matplotlib
    img = mpimg.imread(img_path)
    # Mostrar la imagen en el eje correspondiente
    ax.imshow(img)
    # Desactivar los ejes para una mejor visualización de las imágenes
    ax.axis('off')
    # Agregar texto con la etiqueta debajo de la imagen
    ax.set_title(label, fontsize=16)

# Ajustar automáticamente la disposición de las subtramas para evitar superposiciones
plt.tight_layout()
# Mostrar la figura que contiene las imágenes
plt.show()