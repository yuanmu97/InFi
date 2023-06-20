import numpy as np
import tensorflow as tf 
import os 


def load_and_preprocess_image(img_path):
    img_height = 224
    img_width = 224

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [img_height, img_width])
    img /= 255.0
    return img


def load_and_preprocess_image_pair(img_pair_path):
    img1 = load_and_preprocess_image(img_pair_path[0])
    img2 = load_and_preprocess_image(img_pair_path[1])
    return [img1, img2]


def build_hollywood2_image_dataset(list_path,
                                   root_path="/",
                                   postfix="_30.jpg"):
    img_path_list = list()
    with open(list_path, "r") as fin:
        for line in fin.readlines():
            img_path_list.append(os.path.join(root_path, line.strip()+postfix))
    path_ds = tf.data.Dataset.from_tensor_slices(img_path_list)
    img_ds = path_ds.map(load_and_preprocess_image)

    return img_ds


def build_hollywood2_image_pair_dataset(list_path, root_path="/", postfix="_30.jpg", load_label=True):
    img_pair_path_list = list()
    y = list()
    with open(list_path, "r") as fin:
        for line in fin.readlines():
            tmp = line.strip().split(',')

            p1 = os.path.join(root_path, tmp[0]+postfix)
            p2 = os.path.join(root_path, tmp[1]+postfix)

            img_pair_path_list.append([p1, p2])
            y.append([float(tmp[2])])

    path_ds = tf.data.Dataset.from_tensor_slices(img_pair_path_list)
    img_pair_ds = path_ds.map(load_and_preprocess_image_pair)
    y_ds = tf.data.Dataset.from_tensor_slices(y)

    if not load_label:
        return img_pair_ds

    return tf.data.Dataset.zip((img_pair_ds, y_ds))


def wav2spectrogram(wav_path, 
                    FRAME_LENGTH=255, FRAME_STEP=128, SPEC_HEIGHT=500, SPEC_WIDTH=129):
    wav_binary = tf.io.read_file(wav_path)
    wav_d, _ = tf.audio.decode_wav(wav_binary)
    # only use the first channel
    wav_d = wav_d[:,0]
    spec = tf.signal.stft(wav_d, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
    spec = tf.abs(spec)
    spec = tf.expand_dims(spec, -1)
    spec = tf.image.resize(spec, [SPEC_HEIGHT, SPEC_WIDTH])
    return spec


def load_exec_label(label_path):
    exec_label = np.load(label_path)
    return exec_label


def redundancy_speech(x):
    tmp = list()
    for w in x:
        if w == 1: # <start>
            continue
        elif w == 2: # <end>
            break
        else:
            tmp.append(w)
    l = len(tmp) # the number of recognized words
    return l


def load_wavspec(spec_path):
    spec = np.load(spec_path.numpy())
    return spec


def build_hollywood2_wavspec_dataset(list_path, root_path, label_path=None,
                                     postfix=".npy"):
    def py_func_warpper(spec_path):
        return tf.py_function(load_wavspec, [spec_path], [tf.float32])

    wav_path_list = list()
    with open(list_path, "r") as fin:
        for line in fin.readlines():
            wav_path_list.append(os.path.join(root_path, line.strip()+postfix))
    path_ds = tf.data.Dataset.from_tensor_slices(wav_path_list)
    spec_ds = path_ds.map(py_func_warpper)

    if label_path is None:
        return spec_ds
    
    label_d = np.load(label_path)
    label_ds = tf.data.Dataset.from_tensor_slices(label_d)

    ds = tf.data.Dataset.zip((spec_ds, label_ds))
    return ds