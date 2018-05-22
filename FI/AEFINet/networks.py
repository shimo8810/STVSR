'''
フレーム補間ネットワーク
'''
import chainer
import chainer.links as L
import chainer.functions as F

class BaseAEFINet(chainer.Chain):
    def __init__(self):
        super(BaseAEFINet, self).__init__()

    def __call__(self, x):
        return None

    def get_loss_func(self):
        def lf(x, t):
            y = self(x)
            loss = F.mean_squared_error(y, t)
            self.loss = loss
            self.psnr = 10 * F.log10(1.0 / loss)
            chainer.report({'loss': self.loss, 'psnr':self.psnr}, observer=self)
            return self.loss
        return lf


class AEFINet(BaseAEFINet):
    '''
    AE的に中間層のmapサイズを小さくするネットーワーク
    実験用なのでフィルタサイズやデプス(最終的なmap縮小サイズ)を指定できる
    args:
        f_size:フィルタサイズ(カーネルサイズ), 中間層のフィルタサイズ統一
        ch: チャネル数のパラメータ
    '''
    def __init__(self, f_size=3, ch=2):
        init_w = chainer.initializers.HeNormal()
        n_ch = 8 * ch
        super(AEFINet, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_ch, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv_down2 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_down4  = L.Convolution2D(None, n_ch * 4, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv5 = L.Convolution2D(None, n_ch * 4, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv_up6 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv7 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_up8 = L.Deconvolution2D(None, n_ch, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.conv1(h)) # 8 , H, W
        h  = F.relu(self.conv_down2(h1)) # 16, H/2, W/2
        h2 = F.relu(self.conv3(h)) # 16, H/2, W/2
        h  = F.relu(self.conv_down4(h2)) # 32, H/4, W/4
        h = F.relu(self.conv5(h)) + h # 32, H/4, W/4
        h = F.relu(self.conv_up6(h)) + h2 # 16, H/2, W/2
        h = F.relu(self.conv7(h)) # 16, H/2, W/2
        h = F.relu(self.conv_up8(h)) + h1 # 8 , H, W
        return F.relu(self.conv9(h))

class AEFINetConcat(BaseAEFINet):
    '''
    AE的に中間層のmapサイズを小さくするネットーワーク
    実験用なのでフィルタサイズやデプス(最終的なmap縮小サイズ)を指定できる
    args:
        f_size:フィルタサイズ(カーネルサイズ), 中間層のフィルタサイズ統一
        ch: チャネル数のパラメータ
    '''
    def __init__(self, f_size=3, ch=2):
        init_w = chainer.initializers.HeNormal()
        n_ch = 8 * ch
        super(AEFINetConcat, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_ch, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv_down2 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_down4  = L.Convolution2D(None, n_ch * 4, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv5 = L.Convolution2D(None, n_ch * 4, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv_up6 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv7 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_up8 = L.Deconvolution2D(None, n_ch, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.conv1(h)) # 8 , H, W
        h  = F.relu(self.conv_down2(h1)) # 16, H/2, W/2
        h2 = F.relu(self.conv3(h)) # 16, H/2, W/2
        h  = F.relu(self.conv_down4(h2)) # 32, H/4, W/4
        h = F.concat([F.relu(self.conv5(h)), h], axis=1)  # 32, H/4, W/4
        h = F.concat([F.relu(self.conv_up6(h)), h2], axis=1) # 16, H/2, W/2
        h = F.relu(self.conv7(h)) # 16, H/2, W/2
        h = F.concat([F.relu(self.conv_up8(h)) + h1], axis=1) # 8 , H, W
        return F.relu(self.conv9(h))

class AEFINetBN(BaseAEFINet):
    '''
    AE的に中間層のmapサイズを小さくするネットーワーク
    実験用なのでフィルタサイズやデプス(最終的なmap縮小サイズ)を指定できる
    BNありバージョン
    args:
        f_size:フィルタサイズ(カーネルサイズ), 中間層のフィルタサイズ統一
        ch: チャネル数のパラメータ
    '''
    def __init__(self, f_size=3, ch=2):
        init_w = chainer.initializers.HeNormal()
        n_ch = 8 * ch
        super(AEFINetBN, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_ch, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv_down2 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_down4  = L.Convolution2D(None, n_ch * 4, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv5 = L.Convolution2D(None, n_ch * 4, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv_up6 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv7 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_up8 = L.Deconvolution2D(None, n_ch, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)
            self.norm1 = L.BatchNormalization(n_ch)
            self.norm2 = L.BatchNormalization(n_ch * 2)
            self.norm3 = L.BatchNormalization(n_ch * 2)
            self.norm4 = L.BatchNormalization(n_ch * 4)
            self.norm5 = L.BatchNormalization(n_ch * 4)
            self.norm6 = L.BatchNormalization(n_ch * 2)
            self.norm7 = L.BatchNormalization(n_ch * 2)
            self.norm8 = L.BatchNormalization(n_ch)
            self.norm9 = L.BatchNormalization(3)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.norm1(self.conv1(h))) # 8 , H, W
        h  = F.relu(self.norm2(self.conv_down2(h1))) # 16, H/2, W/2
        h2 = F.relu(self.norm3(self.conv3(h))) # 16, H/2, W/2
        h  = F.relu(self.norm4(self.conv_down4(h2))) # 32, H/4, W/4
        h = F.relu(self.norm5(self.conv5(h))) + h # 32, H/4, W/4
        h = F.relu(self.norm6(self.conv_up6(h))) + h2 # 16, H/2, W/2
        h = F.relu(self.norm7(self.conv7(h))) # 16, H/2, W/2
        h = F.relu(self.norm8(self.conv_up8(h))) + h1 # 8 , H, W
        return F.relu(self.norm9(self.conv9(h)))

class AEFINetConcatBN(BaseAEFINet):
    '''
    AE的に中間層のmapサイズを小さくするネットーワーク
    実験用なのでフィルタサイズやデプス(最終的なmap縮小サイズ)を指定できる
    BNありバージョン
    args:
        f_size:フィルタサイズ(カーネルサイズ), 中間層のフィルタサイズ統一
        ch: チャネル数のパラメータ
    '''
    def __init__(self, f_size=3, ch=2):
        init_w = chainer.initializers.HeNormal()
        n_ch = 8 * ch
        super(AEFINetConcatBN, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_ch, ksize=5, stride=1, pad=2, initialW=init_w)
            self.conv_down2 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv3 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_down4  = L.Convolution2D(None, n_ch * 4, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv5 = L.Convolution2D(None, n_ch * 4, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv_up6 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv7 = L.Convolution2D(None, n_ch * 2, ksize=f_size, stride=1, pad=f_size//2, initialW=init_w)
            self.conv_up8 = L.Deconvolution2D(None, n_ch, ksize=f_size, stride=2, pad=f_size//2, initialW=init_w)
            self.conv9 = L.Convolution2D(None, 3, ksize=5, stride=1, pad=2, initialW=init_w)
            self.norm1 = L.BatchNormalization(n_ch)
            self.norm2 = L.BatchNormalization(n_ch * 2)
            self.norm3 = L.BatchNormalization(n_ch * 2)
            self.norm4 = L.BatchNormalization(n_ch * 4)
            self.norm5 = L.BatchNormalization(n_ch * 4)
            self.norm6 = L.BatchNormalization(n_ch * 2)
            self.norm7 = L.BatchNormalization(n_ch * 2)
            self.norm8 = L.BatchNormalization(n_ch)
            self.norm9 = L.BatchNormalization(3)

    def __call__(self, x):
        h = F.concat((x[:, 0, :, :, :], x[:, 1, :, :, :]), axis=1)
        h1 = F.relu(self.norm1(self.conv1(h))) # 8 , H, W
        h  = F.relu(self.norm2(self.conv_down2(h1))) # 16, H/2, W/2
        h2 = F.relu(self.norm3(self.conv3(h))) # 16, H/2, W/2
        h  = F.relu(self.norm4(self.conv_down4(h2))) # 32, H/4, W/4
        h = F.concat([F.relu(self.norm5(self.conv5(h))), h], axis=1) # 32, H/4, W/4
        h = F.concat([F.relu(self.norm6(self.conv_up6(h))), h2], axis=1) # 16, H/2, W/2
        h = F.relu(self.norm7(self.conv7(h))) # 16, H/2, W/2
        h = F.concat([F.relu(self.norm8(self.conv_up8(h))), h1]) # 8 , H, W
        return F.relu(self.conv9(h))

class AEFINet2(chainer.Chain):
    def __init__(self, f_size=5, n_ch=8, size=64):
        if size % 16 != 0:
            raise ValueError('size must be a multiple of 16.')
        init_w = chainer.initializers.HeNormal()
        super(AEFINet2, self).__init__()

        with self.init_scope():
            # encoder
            self.enc_conv1 = L.Convolution2D(None, n_ch * 1, ksize=f_size,
                                stride=1, pad=f_size//2, initialW=init_w) # 64x64
            self.enc_conv2 = L.Convolution2D(None, n_ch * 2, ksize=f_size,
                                stride=2, pad=f_size//2, initialW=init_w) # 32x32
            self.enc_conv3 = L.Convolution2D(None, n_ch * 4, ksize=f_size,
                                stride=2, pad=f_size//2, initialW=init_w) # 16x16
            self.enc_conv4 = L.Convolution2D(None, n_ch * 8, ksize=f_size,
                                stride=2, pad=f_size//2, initialW=init_w) # 8x8

            # decoder
            self.dec_conv1 = L.Deconvolution2D(None, n_ch * 4, ksize=f_size,
                    stride=2, pad=f_size//2, initialW=init_w, outsize=(size//4, size//4)) # 16x16
            self.dec_conv2 = L.Deconvolution2D(None, n_ch * 2, ksize=f_size,
                    stride=2, pad=f_size//2, initialW=init_w, outsize=(size//2, size//2)) # 32z32
            self.dec_conv3 = L.Deconvolution2D(None, n_ch * 1, ksize=f_size,
                    stride=2, pad=f_size//2, initialW=init_w, outsize=(size, size)) # 64x64
            self.dec_conv4 = L.Convolution2D(None, 3, ksize=f_size,
                    stride=1, pad=f_size//2, initialW=init_w) # 64x64

    def encode(self, x):
        batch, f, c, w, h = x.shape
        h = F.reshape(x, (batch, f * c, w, h))
        h1 = F.relu(self.enc_conv1(h)) # 64x64
        h2 = F.relu(self.enc_conv2(h1)) # 32z32
        h3 = F.relu(self.enc_conv3(h2)) # 16x16
        h4 = F.relu(self.enc_conv4(h3)) # 8x8
        return h1, h2, h3, h4

    def decode(self, h1, h2, h3, h4):
        h = F.concat((F.relu(self.dec_conv1(h4)), h3), axis=1)
        h = F.concat((F.relu(self.dec_conv2(h)),  h2), axis=1)
        h = F.concat((F.relu(self.dec_conv3(h)),  h1), axis=1)
        return F.relu(self.dec_conv4(h))

    def __call__(self, x):
        h1, h2, h3, h4 = self.encode(x)
        return self.decode(h1, h2, h3, h4)

    def get_loss_func(self):
        def lf(x, t):
            h1, h2, h3, h4 = self.encode(x)
            y = self.decode(h1, h2, h3, h4)
            loss = F.mean_squared_error(y, t)
            self.loss = loss
            self.psnr = 10 * F.log10(1.0 / loss)
            chainer.report({'loss': self.loss, 'psnr':self.psnr}, observer=self)
            return self.loss
        return lf
