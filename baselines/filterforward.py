"""
FilterForward 
reference: 
    https://github.com/viscloud/ff
"""
import tensorflow as tf 
from tensorflow.keras import layers
import time
import os 
from pathlib import Path


def build_mc_full_frame():
    mc = tf.keras.models.Sequential([
        layers.Conv2D(32, (1,1), strides=(1,1), activation='relu'),
        layers.Conv2D(32, (1,1), strides=(1,1), activation='relu'),
        layers.Conv2D(1, (1,1), strides=(1,1), activation='relu'),
        layers.GlobalMaxPooling2D(),
        layers.Activation('sigmoid')
    ])
    return mc


def build_mc_localized():
    mc = tf.keras.models.Sequential([
        layers.SeparableConv2D(16, (3,3), strides=(1,1), activation='relu'),
        layers.SeparableConv2D(32, (3,3), strides=(2,2), activation='relu'),
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return mc 


def build_mc_windowed(frame_n):
    input_list = list()
    for idx in range(frame_n):
        input_layer = layers.Input(shape=(224, 224, 3))
        input_list.append(input_layer)
    
    x_list = list()
    for inp in input_list:
        x = layers.Conv2D(32, (1,1), strides=(1,1))(inp)
        x_list.append(x)
    
    x = layers.Concatenate(axis=-1)(x_list)
    
    x = layers.Conv2D(32, (3,3), strides=(1,1), activation='relu')(x)
    x = layers.Conv2D(32, (3,3), strides=(2,2), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    mc = tf.keras.Model(input_list, x)
    return mc


# FF configuration files
# https://github.com/viscloud/ff/tree/master/scripts/e2e/configs/ff_confs

def train_mc(mc, x, y,
             learning_rate=0.01,
             batch_size=None,
             epochs=1,
             x_test=None, y_test=None,
             log_dir=None,
             weight_dir=None):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    mc.compile(optimizer=optimizer,
               loss="binary_crossentropy",
               metrics=["binary_accuracy"])
    
    callbacks = list()
    if log_dir is not None:
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks.append(tb_callback)
    if weight_dir is not None:
        Path(weight_dir).mkdir(parents=True, exist_ok=True)
        timestr = time.strftime("%Y%m%d-%H%M%S-")
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(weight_dir, timestr+"BEST.h5"),
            monitor="val_binary_accuracy",
            mode="max",
            save_best_only=True
        )
        callbacks.append(ckpt_callback)
    
    if x_test is not None and y_test is not None:
        validation_data = (x_test, y_test)
    else:
        validation_data = None

    res = mc.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                 validation_data=validation_data, verbose=1)
    
    if weight_dir is not None:
        mc.save(os.path.join(weight_dir, timestr+f"Epoch{epochs}.h5"))
    
    return res
    
