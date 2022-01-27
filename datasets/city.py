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


def build_city_image_dataset(list_path, 
                             root_path="/",
                             start=None, n=None):
    img_path_list = list()
    with open(list_path, "r") as fin:
        for line in fin.readlines():
            img_path_list.append(os.path.join(root_path, line.strip()))
    if start is not None and n is not None:
        img_path_list = img_path_list[start:start+n]

    path_ds = tf.data.Dataset.from_tensor_slices(img_path_list)
    img_ds = path_ds.map(load_and_preprocess_image)

    return img_ds


def build_city_image_pair_dataset(list_path, 
                                  root_path="/"):
    img_pair_path_list = list()
    y = list()
    with open(list_path, "r") as fin:
        for line in fin.readlines():
            tmp = line.strip().split(',')

            p1 = os.path.join(root_path, tmp[0])
            p2 = os.path.join(root_path, tmp[1])

            img_pair_path_list.append([p1, p2])
            y.append([float(tmp[2])])

    path_ds = tf.data.Dataset.from_tensor_slices(img_pair_path_list)
    img_pair_ds = path_ds.map(load_and_preprocess_image_pair)
    y_ds = tf.data.Dataset.from_tensor_slices(y)

    return tf.data.Dataset.zip((img_pair_ds, y_ds))


def build_vc_mp_pair_dataset(root_path, split='train'):
    x = np.load(os.path.join(root_path, f"city_vc_mp_reuse_{split}_x_pair.npy"))

    x1 = x[:,0]
    x2 = x[:,1]

    y = np.load(os.path.join(root_path, f"city_vc_mp_reuse_{split}_sim.npy"))

    def generator():
        for s1, s2, l in zip(x1, x2, y):
            yield {"input-1": s1, "input-2": s2}, l 

    ds = tf.data.Dataset.from_generator(generator, output_types=({"input-1": tf.float32, "input-2": tf.float32}, tf.float32))
    return ds


def load_exec_label(label_path):
    exec_label = np.load(label_path)
    return exec_label
