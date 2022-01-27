"""
InFi-Skip functions
"""
import tensorflow as tf 
from tensorflow.keras import layers
from .modality_nn import build_image_nn, build_video_nn, build_audio_nn, build_text_nn, build_vec_nn
import time
import os
from pathlib import Path


def build_classifier(x, n_dense, dropout):
    """
    binary classifier for making SKIP decision

    Args:
        x (Tensor)
        n_dense (int): the number of dense units
        dropout (float): dropout ratio
    
    Returns:
        out (Tensor)
    """
    x = layers.Dense(n_dense, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return out


def build_infi_skip(modality, input_shape, n_dense, dropout=0.5,
                    n_layers=None, n_filters=None, n_frames=None,
                    vocab_size=None, emb_len=None):
    """
    build InFi-Skip model

    Args:
        modality (string): in ["image", "video", "audio", "text", "vector"]
        input_shape (tuple of int)
        n_dense (int): the number of dense units
        dropout (float): dropout ratio in classifier
        n_layers (int): [image, video, audio] the number of conv layers
        n_filters (int): [image, video, audio] the number of conv filters
        n_frames (int): [video] the number of stacked frames
        vocab_size (int): [text] the size of vocabulary
        emb_len (int): [text, vector] the length of embedding

    Returns:
        infi_skip (Model): tf.keras.Model instance
    """
    assert(modality in ["image", "video", "audio", "text", "vector"])
    if modality == "image":
        inp, x = build_image_nn(input_shape, n_filters, n_layers)
    elif modality == "video":
        inp, x = build_video_nn(input_shape, n_frames, n_filters, n_layers)
    elif modality == "audio":
        inp, x = build_audio_nn(input_shape, n_filters, n_layers)
    elif modality == "text":
        inp, x = build_text_nn(input_shape, vocab_size, emb_len)
    elif modality == "vector":
        inp, x = build_vec_nn(input_shape, emb_len)

    out = build_classifier(x, n_dense, dropout)
    infi_skip = tf.keras.Model(inp, out)
    return infi_skip


def train_infi_skip(infi_skip, train_data, 
                    learning_rate=0.001, batch_size=32, epochs=20,
                    val_data=None, log_dir=None, weight_dir=None,
                    optimizer=None, loss="binary_crossentropy"):
    """
    train InFi-Skip model

    Args:
        infi_skip (Model): InFi-Skip model
        train_data (tf.data.Dataset): training dataset
        learning_rate (float)
        batch_size (int)
        epochs (int)
        val_data (tf.data.Dataset): validation dataset
        log_dir (str): /path/to/tensorboard-log/
        weight_dir (str): /path/to/weight/
    Returns:
        res (History)
    """
    if optimizer is None:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    infi_skip.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])
    
    callbacks = list()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, timestr))
        callbacks.append(tb_callback)
    if weight_dir is not None:
        Path(weight_dir).mkdir(parents=True, exist_ok=True)
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(weight_dir, timestr+"-BEST.h5"),
            monitor="val_binary_accuracy",
            mode="max",
            save_best_only=True
        )
        callbacks.append(ckpt_callback)
    if val_data is not None:
        validation_data = val_data.batch(batch_size)
    else:
        validation_data = None
    train_d = train_data.batch(batch_size)
    res = infi_skip.fit(train_d, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                        validation_data=validation_data, verbose=1)
    
    if weight_dir is not None:
        infi_skip.save(os.path.join(weight_dir, timestr+f"-Epoch{epochs}.h5"))
    
    return res


def load_infi_skip(weight_path):
    """
    load InFi-Skip model checkpoint

    Args:
        weight_path (str): /path/to/checkpoint.h5
    
    Returns:
        m (Model): InFi-Skip model
    """
    m = tf.keras.models.load_model(weight_path)
    return m