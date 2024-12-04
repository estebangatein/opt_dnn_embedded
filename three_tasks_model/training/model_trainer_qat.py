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


from tensorflow.keras.utils import load_img, img_to_array

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)




def resnet8(img_width=200, img_height=200, img_channels=1, output_dim_coll=2, output_dim_objects=4, output_dim_steer=1, summary=False):
    """
    Define model architecture without skip connections (Residual connections removed).
    
    # Arguments
       img_width: Target image width.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim_coll: Output dimension for collision prediction.
       output_dim_objects: Output dimension for object detection.
       output_dim_steer: Output dimension for steering prediction.
       
    # Returns
       model: A quantized-aware Model instance.
    """

    # Input layer
    img_input = keras.layers.Input(shape=(img_height, img_width, img_channels))

    # First convolutional block
    x = keras.layers.Conv2D(32, (5, 5), strides=[1, 1], padding='same')(img_input)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=[2, 2])(x)
    
    # First block (Conv + Activation + Conv)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=[1, 1], padding='same',
                            kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(32, (3, 3), padding='same',
                            kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    # MaxPooling and Dropout for object output
    obj_branch = keras.layers.Activation('relu')(x)
    obj_branch = keras.layers.MaxPooling2D(pool_size=(2, 2))(obj_branch)
    obj_branch = keras.layers.Flatten()(obj_branch)
    obj_branch = keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))(obj_branch)
    obj_branch = keras.layers.Activation('relu')(obj_branch)
    obj_branch = keras.layers.Dropout(0.3)(obj_branch)

    # Output for objects
    objects = keras.layers.Dense(output_dim_objects, activation='softmax', name='fire_output')(obj_branch)

    # Second block (Conv + Activation + Conv)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=[2, 2], padding='same',
                            kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same',
                            kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    # Third block (Conv + Activation + Conv)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=[2, 2], padding='same',
                            kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same',
                            kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    # Flatten and apply Dropout
    x = keras.layers.Flatten()(x)
    # x = keras.layers.Activation('relu')(x)
    # x = keras.layers.Dropout(0.5)(x)

    # Steering output
    # steer = keras.layers.Dense(output_dim_steer, name='steering_output')(x)

    # Collision output
    coll = keras.layers.Dense(output_dim_coll, activation='softmax', name='labels_output')(x)
    steer = keras.layers.Dense(output_dim_steer, name='steering_output')(x)

    # Define model with multiple outputs (collision, objects, steering)
    model = keras.models.Model(inputs=[img_input], outputs=[coll, objects, steer])

    # Apply quantization-aware training
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)

    if summary:
        print(q_aware_model.summary())

    return q_aware_model


model = resnet8()

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

images_train, labels_train = dataset_creator('data/merged_data/training', 100)
indices = np.random.permutation(images_train.shape[0])
images_train = images_train[indices]
labels_train = labels_train[indices]
y_col_train, y_fire_train, y_steer_train = labels_train[:,0, :][:, :2], labels_train[:, 1, :], labels_train[:, 2, :][:, 0]

images_val, labels_val = dataset_creator('data/merged_data/validation', 5)
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


sample_weight_task1_train = np.logical_not(np.isnan(np.sum(y_col_train, axis=1))).astype(np.float32)  # Ponderaciones para task1
sample_weight_task2_train = np.logical_not(np.isnan(np.sum(y_fire_train, axis=1))).astype(np.float32)  # Ponderaciones para task2
sample_weight_task3_train = np.logical_not(np.isnan(y_steer_train)).astype(np.float32)  # Ponderaciones para task3

# Rellenar NaN con algún valor temporal, pero solo para evitar errores (los pesos sample_weight manejarán NaNs)
y_train_task1 = np.nan_to_num(y_col_train, nan=999)
y_train_task2 = np.nan_to_num(y_fire_train, nan=999)
y_train_task3 = np.nan_to_num(y_steer_train, nan=999)


sample_weight_task1_val = np.logical_not(np.isnan(np.sum(y_col_val, axis=1))).astype(np.float32)  # Ponderaciones para task1
sample_weight_task2_val = np.logical_not(np.isnan(np.sum(y_fire_val, axis=1))).astype(np.float32)  # Ponderaciones para task2
sample_weight_task3_val = np.logical_not(np.isnan(y_steer_val)).astype(np.float32)  # Ponderaciones para task3

# Rellenar NaN con algún valor temporal, pero solo para evitar errores (los pesos sample_weight manejarán NaNs)
y_val_task1 = np.nan_to_num(y_col_val, nan=999)
y_val_task2 = np.nan_to_num(y_fire_val, nan=999)
y_val_task3 = np.nan_to_num(y_steer_val, nan=999)



class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()

class CustomEarlyStopping(Callback):
    def __init__(self, patience=0, min_delta=0, start_epoch=1):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch
        self.wait = 0
        self.best_weights = None
        self.best = float('inf')  # Start with a high value for comparison
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # No hacer nada si estamos en las primeras épocas
        if epoch < self.start_epoch:
            return

        current_loss = logs.get("val_loss")
        if current_loss is None:
            return

        # Si la pérdida ha mejorado (por lo menos min_delta)
        if current_loss < self.best - self.min_delta:
            self.best = current_loss
            self.best_epoch = epoch + 1
            self.wait = 0
            self.best_weights = self.model.get_weights()  # Guardar los mejores pesos
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Restaurar los mejores pesos
                self.model.stop_training = True
                print(f"Restaurando pesos de la mejor época: {self.best_epoch} con val_loss: {self.best}")
                self.model.set_weights(self.best_weights)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
early_stopping = CustomEarlyStopping(patience=15, min_delta=0.001, start_epoch=1)


model.compile(optimizer='adam',
              loss={'quant_labels_output': custom_categorical_crossentropy_loss, 'quant_fire_output':custom_categorical_crossentropy_loss, 'quant_steering_output': custom_mse_loss}, weighted_metrics = {'quant_labels_output': 'accuracy', 'quant_fire_output':'accuracy', 'quant_steering_output': 'mse'})


history = model.fit(x = images_train, 
          y = {'quant_labels_output':y_train_task1, 'quant_fire_output':y_train_task2, 'quant_steering_output': y_train_task3}, 
          sample_weight={'quant_labels_output':sample_weight_task1_train, 'quant_fire_output':sample_weight_task2_train, 'quant_steering_output': sample_weight_task3_train}, 
          epochs=100, 
          batch_size=32, 
          validation_data = (images_val, 
                             {'quant_labels_output':y_val_task1, 'quant_fire_output':y_val_task2, 'quant_steering_output': y_val_task3}, 
                             {'quant_labels_output':sample_weight_task1_val, 'quant_fire_output':sample_weight_task2_val, 'quant_steering_output': sample_weight_task3_val}), 
          validation_batch_size=128,
          callbacks = [ClearMemory(), reduce_lr, early_stopping])

with open('training_metrics.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('qat_model.tflite', 'wb') as f:
    f.write(tflite_model)


end_time = datetime.now()
print(f"Fin del script: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

