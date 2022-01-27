"""
neural networks used in InFi
"""

import tensorflow as tf
from tensorflow.keras import layers


def conv_step(inp, f, h, w, s=1):
    a1 = layers.Activation("relu")(inp)
    c1 = layers.SeparableConv2D(f, (h, w), strides=s, dilation_rate=1, padding="same")(a1)
    ln1 = layers.LayerNormalization()(c1)
    return ln1


def conv_res(inp, F):
    c1 = conv_step(inp, f=F, h=3, w=3)
    c2 = conv_step(c1, f=F, h=3, w=3)
    p1 = layers.MaxPool2D((3,3), strides=2, padding="same")(c2)
    res = conv_step(inp, f=F, h=1, w=1, s=2)
    out = layers.Add()([p1, res])
    return out


def build_image_nn(input_shape, n_filters, n_layers):
    """
    feature network for image modality

    Args:
        input_shape
        n_filters (int): the number of conv filters
        n_layers (int): the number of conv layers
    
    Returns:
        inp: keras input layer
        x: resulting tensor
    """
    inp = layers.Input(shape=input_shape)
    F = n_filters
    x = conv_res(inp, F)
    for _ in range(n_layers-1):
        F *= 2
        x = conv_res(x, F)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    return inp, x


def build_video_nn(input_shape, n_frames, n_filters, n_layers):
    """
    feature network for video (stacked images) modality

    Args:
        input_shape
        n_frames (int): the number of stacked frames
        n_filters (int): the number of conv filters
        n_layers (int): the number of conv layers
    
    Returns:
        inp_list: a list of keras input layers
        x: resulting tensor
    """
    inp_list = list()
    for _ in range(n_frames):
        inp = layers.Input(shape=input_shape)
        inp_list.append(inp)
    
    x_list = list()
    for inp in inp_list:
        F = n_filters
        x = conv_res(inp, F)
        for _ in range(n_layers-1):
            F *= 2
            x = conv_res(x, F)
        x_list.append(x)
    x = layers.Concatenate(axis=-1)(x_list)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    return inp_list, x


def build_audio_nn(input_shape, n_filters, n_layers):
    """
    feature network for audio (specturm) modality

    Args:
        input_shape
        n_filters (int): the number of conv filters
        n_layers (int): the number of conv layers

    Returns:
        inp: keras input layer
        x: resulting tensor
    """
    inp = layers.Input(shape=input_shape)
    F = n_filters
    x = conv_res(inp, F)
    for _ in range(n_layers-1):
        F *= 2
        x = conv_res(x, F)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    return inp, x


def build_text_nn(input_shape, vocab_size, emb_len):
    """
    feature network for text (word index seq) modality

    Args:
        input_shape
        vocab_size (int): the size of vocabulary
        emb_len (int): the length of embedding
    
    Returns:
        inp: keras input layer
        x: resulting tensor
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Embedding(vocab_size, emb_len)(inp)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Flatten()(x)
    return inp, x


def build_vec_nn(input_shape, emb_len):
    """
    feature network for vector modality

    Args:
        input_shape
        emb_len (int): the length of embedding

    Returns:
        inp: keras input layer
        x: resulting tensor
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Dense(emb_len)(inp)
    x = layers.Flatten()(x)
    return inp, x