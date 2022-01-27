"""
InFi-Reuse functions
"""
import tensorflow as tf 
from tensorflow.keras import layers
from .modality_nn import build_image_nn, build_video_nn, build_audio_nn, build_text_nn, build_vec_nn
import time
from pathlib import Path
import os 


def build_infi_reuse(modality, input_shape, n_dense, dropout=0.5,
                    n_layers=None, n_filters=None, n_frames=None,
                    vocab_size=None, emb_len=None):
    """
    build InFi-Reuse model

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
        infi_reuse (Model): tf.keras.Model instance
    """
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

    x = layers.Dense(n_dense, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    feat_nn = tf.keras.Model(inp, x)

    inp1 = layers.Input(shape=input_shape, name="input-1")
    inp2 = layers.Input(shape=input_shape, name="input-2")
    emb1 = feat_nn(inp1)
    emb2 = feat_nn(inp2)
    out = tf.math.abs(emb1-emb2)
    out = layers.Dense(1, activation='sigmoid')(out)

    infi_reuse = tf.keras.Model(inputs=[inp1, inp2], outputs=out)
    return infi_reuse


def siamese_nn_loss(margin=1):
    """
    Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Args:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """
        Calculates the constrastive loss.

        Args:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true, each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


def train_infi_reuse(infi_reuse, train_data, 
                    learning_rate=0.001, batch_size=32, epochs=20,
                    val_data=None, log_dir=None, weight_dir=None):
    """
    train InFi-Reuse model

    Args:
        infi_reuse (Model): InFi-Reuse model
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
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    infi_reuse.compile(optimizer=optimizer, loss=siamese_nn_loss(margin=1), metrics=["binary_accuracy"])
    
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
    res = infi_reuse.fit(train_d, batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                        validation_data=validation_data, verbose=1)
    
    if weight_dir is not None:
        infi_reuse.save(os.path.join(weight_dir, timestr+f"-Epoch{epochs}.h5"))
    
    return res


def load_infi_reuse(weight_path):
    """
    load InFi-Reuse model checkpoint

    Args:
        weight_path (str): /path/to/checkpoint.h5
    
    Returns:
        m (Model): InFi-Reuse model
    """
    m = tf.keras.models.load_model(weight_path, 
                                   custom_objects={"tf": tf, "contrastive_loss": siamese_nn_loss()})
    return m