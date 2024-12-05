import pickle
import os
from datetime import datetime
import tensorflow as tf 
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import custom_object_scope
import pandas as pd
from tensorflow.keras.utils import load_img, img_to_array
import time
"""
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

"""


def dataset_creator(path_to_folder, max_by_experiment = 3):
    col, fire, steer = 0, 0, 0
    images, labels = [], []

    for folder in os.listdir(path_to_folder):

        if os.path.isdir(os.path.join(path_to_folder, folder)):

            for file in os.listdir(os.path.join(path_to_folder, folder)):

                if file.endswith('.txt'):
                    if 'labels' in file and col < max_by_experiment:
                        col += 1
                        for pic in sorted(os.listdir(os.path.join(path_to_folder, folder, 'images'))):
                            img = load_img(os.path.join(os.path.join(path_to_folder, folder, 'images'), pic), target_size=(200, 200), color_mode='grayscale')  # Ajustar tamaño y modo de color
                            img_array = img_to_array(img) / 255.0  # Normalizar la imagen
                            images.append(img_array)
                        
                        labels_txt = np.loadtxt(os.path.join(path_to_folder, folder, file))

                        for label in labels_txt:
                            if label == 0:
                                label = [np.array([1, 0]), np.array([np.nan]*4), np.array([np.nan])]
                            elif label == 1:
                                label = [np.array([0, 1]), np.array([np.nan]*4), np.array([np.nan])]
                            label = pad_sequences(label, dtype='float32', padding='post', value=np.nan)
                            labels.append(label)


                    elif 'fire' in file and fire < max_by_experiment:
                        fire += 1
                        for pic in sorted(os.listdir(os.path.join(path_to_folder, folder, 'images'))):
                            img = load_img(os.path.join(os.path.join(path_to_folder, folder, 'images'), pic), target_size=(200, 200), color_mode='grayscale')  # Ajustar tamaño y modo de color
                            img_array = img_to_array(img) / 255.0  # Normalizar la imagen
                            images.append(img_array)
                        
                        labels_txt = np.loadtxt(os.path.join(path_to_folder, folder, file), delimiter=' ')

                        for label in labels_txt:
                            label = [np.array([np.nan]*2), label, np.array([np.nan])]
                            label = pad_sequences(label, dtype='float32', padding='post', value=np.nan)
                            labels.append(label)

                            
                    elif 'sync' in file and steer < max_by_experiment:
                        steer += 1
                        for pic in sorted(os.listdir(os.path.join(path_to_folder, folder, 'images'))):
                            img = load_img(os.path.join(os.path.join(path_to_folder, folder, 'images'), pic), target_size=(200, 200), color_mode='grayscale')
                            img_array = img_to_array(img) / 255.0
                            images.append(img_array)

                        labels_txt = np.loadtxt(os.path.join(path_to_folder, folder, file), usecols=0, delimiter=',', skiprows=1)

                        for label in labels_txt:
                            label = [np.array([np.nan]*2), np.array([np.nan]*4), np.array([label])]
                            label = pad_sequences(label, dtype='float32', padding='post', value=np.nan)
                            labels.append(label)

    return np.array(images), np.array(labels)

images_val, labels_val = dataset_creator('../../testing_data', 5)
indices = np.random.permutation(images_val.shape[0])
images_val = images_val[indices]
labels_val = labels_val[indices]
y_col_val, y_fire_val, y_steer_val = labels_val[:,0, :][:, :2], labels_val[:, 1, :], labels_val[:, 2, :][:, 0]

def custom_loss(y_true, y_pred, weights=[1.0, 1.0, 1.0]):
    total_loss = 0.0
    valid_tasks = 0

    tf.print(y_pred)

    # Clasificación binaria
    if y_true[0] is not np.nan:  # Si hay etiqueta para la tarea binaria
        binary_loss = tf.keras.losses.binary_crossentropy(y_true[0], y_pred[0])
        total_loss += weights[0] * binary_loss
        valid_tasks += 1

    # Clasificación multiclase
    if y_true[1][0] is not np.nan:  # Si hay etiqueta para la tarea multiclase
        categorical_loss = tf.keras.losses.categorical_crossentropy(y_true[1], y_pred[1])
        total_loss += weights[1] * categorical_loss
        valid_tasks += 1

    # Regresión
    if y_true[2] is not np.nan:  # Si hay etiqueta para la tarea de regresión
        regression_loss = tf.keras.losses.MSE(y_true[2], y_pred[2])
        total_loss += weights[2] * regression_loss
        valid_tasks += 1
    # Evitar la división por cero
    if valid_tasks > 0:
        tf.print(total_loss / valid_tasks)  
        return total_loss / valid_tasks
    else:
        print('error')
        return tf.constant(0.0)  # O un valor pequeño si no hay tareas válidas


def custom_mse_loss(y_true, y_pred):
    # Ignorar el valor simbólico 999
    mask = tf.not_equal(y_true, 999)

    # Aplicar la máscara
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Si no hay datos válidos, retornar un valor grande
    if tf.shape(y_true_masked)[0] == 0:
        return tf.constant(1e6)  # Valor grande para indicar error

    # Calcular la pérdida utilizando operaciones de TensorFlow
    loss = tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))
    return loss



def custom_binary_crossentropy_loss(y_true, y_pred):
    # Ignorar el valor simbólico 999
    mask = tf.not_equal(y_true, 999)
    
    # Aplicar la máscara
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Si no hay datos válidos, retornar un valor grande
    if tf.shape(y_true_masked)[0] == 0:
        return tf.constant(1e6)  # Valor grande para indicar error

    # Calcular la pérdida utilizando operaciones de TensorFlow
    loss = -tf.reduce_mean(y_true_masked * tf.math.log(y_pred_masked + 1e-6) +
                           (1 - y_true_masked) * tf.math.log(1 - y_pred_masked + 1e-6))
    return loss



def custom_categorical_crossentropy_loss(y_true, y_pred):
    # Ignorar el valor simbólico 999
    mask = tf.not_equal(y_true, 999)
    
    # Aplicar la máscara
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Si no hay datos válidos, retornar un valor grande
    if tf.shape(y_true_masked)[0] == 0:
        return tf.constant(1e6)  # Valor grande para indicar error

    # Calcular la pérdida utilizando operaciones de TensorFlow
    loss = -tf.reduce_sum(y_true_masked * tf.math.log(y_pred_masked + 1e-15)) / tf.cast(tf.shape(y_true_masked)[0], tf.float32)
    return loss

from keras.layers import TFSMLayer

model = TFSMLayer('original_model', call_endpoint='serving_default')

"""with custom_object_scope({'custom_loss':custom_loss, 'custom_mse_loss':custom_mse_loss, 'custom_categorical_crossentropy_loss':custom_categorical_crossentropy_loss, 'custom_binary_crossentropy_loss':custom_binary_crossentropy_loss}):
    model = tf.keras.models.load_model('original_model', compile=False)
"""

# Función para medir el tiempo promedio de inferencia para un modelo normal
def measure_inference_time_tf(model, images, n_iterations=10):
    total_time = 0
    predictions = []

    # Iterar sobre las imágenes
    for i, img in enumerate(images):
        start_time = time.time()
        pred = model(tf.expand_dims(img, axis=0))  # Expandir dimensión para batch de tamaño 1
        total_time += time.time() - start_time
        predictions.append(pred)

    avg_time = total_time / len(images)
    return avg_time, predictions

# Función para medir el tiempo promedio de inferencia para un modelo TFLite
def measure_inference_time_tflite(tflite_path, images):
    # Cargar el modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Obtener detalles de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    total_time = 0
    predictions = []

    # Iterar sobre las imágenes
    for img in images:
        # Preparar entrada
        input_data = tf.expand_dims(img, axis=0).numpy().astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()  # Ejecutar inferencia
        total_time += time.time() - start_time

        # Extraer predicciones
        outputs = [interpreter.get_tensor(output['index']) for output in output_details]
        predictions.append(outputs)

    avg_time = total_time / len(images)
    return avg_time, predictions

# Código principal
# Medir tiempos de inferencia con el modelo original
model_path_tflite = 'qat_model.tflite'  # Sustituye por el path al modelo TFLite

print("Midiendo tiempos de inferencia para el modelo TensorFlow...")
original_avg_time, predictions_tf = measure_inference_time_tf(model, images_val[:100])
print('original average time', original_avg_time)

print("Midiendo tiempos de inferencia para el modelo TFLite...")
tflite_avg_time, predictions_tflite = measure_inference_time_tflite(model_path_tflite, images_val[:100])
print('qat average time', tflite_avg_time)
