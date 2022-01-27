import numpy as np
import tensorflow as tf 
import os 


def build_mocap_pair_dataset(root_path, split='train'):
    x = np.load(os.path.join(root_path, f"x_pair_{split}.npy"))
    N = len(x)

    x1 = x[:,0,:]
    x2 = x[:,1,:]

    y = np.load(os.path.join(root_path, f"y_reuse_{split}.npy"))

    def generator():
        for s1, s2, l in zip(x1, x2, y):
            yield {"input-1": s1, "input-2": s2}, l 

    ds = tf.data.Dataset.from_generator(generator, output_types=({"input-1": tf.float32, "input-2": tf.float32}, tf.float32))
    return ds