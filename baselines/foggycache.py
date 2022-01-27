import cv2
import numpy as np
import librosa
import tqdm
from sklearn.neighbors import KNeighborsClassifier


def compute_sift(img_path_list):
    sift = cv2.SIFT_create()
    res = [] 
    for img_path in tqdm.tqdm(img_path_list):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        des = (np.mean(des, axis=0).astype('float32') if des is not None else np.zeros(128))
        res.append(des)
    
    res = np.array(res)
    print(res.shape)
    return res


def compute_mfcc(wav_path_list, des_len=128):
    res = []
    for wav_path in tqdm.tqdm(wav_path_list):
        y, sr = librosa.load(wav_path)
        des = librosa.feature.mfcc(y=y, sr=sr)
        des = np.mean(des, axis=0).astype('float32')
        des = np.resize(des, des_len)
        res.append(des)
    
    res = np.array(res)
    print(res.shape)
    return res


class LSHTable(object):
    
    def __init__(self, dim, n_func, n_neighbors=10):
        self.dim = dim
        self.n_func = n_func
        self.w_list, self.b_list = self.init_lsh_func()
        self.X = []
        self.Y = []
        
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
    def init_lsh_func(self):
        mu, sigma = 0, 1
        w_list = []
        b_list = []
        for _ in range(self.n_func):
            b_list.append(np.random.uniform(0, 1))
            w_list.append(np.random.normal(mu, sigma, self.dim))
        return w_list, b_list
    
    def lsh(self, k):
        x = []
        for w, b in zip(self.w_list, self.b_list):
            x.append(np.floor(np.dot(w, k) + b))
        return x
    
    def push(self, k, val):
        self.X.append(self.lsh(k))
        self.Y.append(val)

    
    def train(self):
        self.knn.fit(self.X, self.Y)

        
    def predict(self, k):
        x = self.lsh(k)
        y = self.knn.predict([x])
        return y