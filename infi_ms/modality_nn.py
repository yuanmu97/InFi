from mindspore import nn, ops


class ConvStep(nn.Cell):
    def __init__(self, in_channels, f, h, w, ln_shape, s=1):
        super(ConvStep, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=f, kernel_size=(h, w), stride=s)
        self.ln = nn.LayerNorm(ln_shape, begin_norm_axis=1, begin_params_axis=1)

    def construct(self, inp):
        a1 = self.relu(inp)
        c1 = self.conv(a1)
        ln1 = self.ln(c1)
        return ln1
    

class ConvRes(nn.Cell):
    def __init__(self, in_channels, F, ln_h, ln_w):
        super(ConvRes, self).__init__()
        self.conv_step1 = ConvStep(in_channels, F, 3, 3, (F, ln_h, ln_w))
        self.conv_step2 = ConvStep(F, F, 3, 3, (F, ln_h, ln_w))
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.conv_step3 = ConvStep(in_channels, F, 1, 1, (F, ln_h//2, ln_w//2), s=2)
    
    
    def construct(self, inp):
        c1 = self.conv_step1(inp)
        c2 = self.conv_step2(c1)
        p1 = self.max_pool2d(c2)
        res = self.conv_step3(inp)
        out = ops.add(p1, res)
        return out


class ImageNN(nn.Cell):
    def __init__(self, input_shape, n_filters):
        super(ImageNN, self).__init__()

        inp_c, inp_h, inp_w = input_shape
        self.conv_res1 = ConvRes(inp_c, n_filters, inp_h, inp_w)
        self.conv_res2 = ConvRes(n_filters, n_filters*2, inp_h//2, inp_w//2)
        self.global_maxpool = nn.MaxPool2d(kernel_size=inp_h//4)

    def construct(self, inp):
        x = self.conv_res1(inp)
        x = self.conv_res2(x)
        x = self.global_maxpool(x)
        x = ops.squeeze(x, axis=(-1, -2))
        return x


if __name__ == "__main__":
    import mindspore as ms
    import numpy as np

    net = ImageNN((3, 224, 224), 16)
    x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
    y = net(x)
    print(y.shape)