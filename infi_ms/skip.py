from mindspore import nn, ops
from modality_nn import ImageNN


class Classifier(nn.Cell):
    def __init__(self, in_channels, n_dense, dropout):
        super(Classifier, self).__init__()
        self.dense = nn.Dense(in_channels, n_dense, has_bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.dense2 = nn.Dense(n_dense, 1, has_bias=True)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x 
    

class InFiSkip(nn.Cell):
    def __init__(self, modality, input_shape, n_dense, dropout, n_filters):
        super(InFiSkip, self).__init__()
        if modality == "image":
            self.modality_nn = ImageNN(input_shape, n_filters)
            inp_c, inp_h, inp_w = input_shape
            self.clf = Classifier(in_channels=inp_h//4, n_dense=n_dense, dropout=dropout)

    def construct(self, x):
        x = self.modality_nn(x)
        x = clf(x)
        return x
    


if __name__ == "__main__":
    import mindspore as ms
    import numpy as np

    net = ImageNN((3, 224, 224), 16)
    x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
    y = net(x)
    print(y.shape)


    clf = Classifier(32, 128, 0.5)
    x2 = ms.Tensor(np.ones([1, 32]), ms.float32)
    y2 = clf(x2)
    print(y2.shape)


    infi = InFiSkip("image", (3, 224, 224), 128, 0.5, 16)
    x3 = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
    y3 = infi(x3)
    print(y3.shape)