import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import (reporter, training)
from chainer.training import extensions
from chainer.datasets import (TupleDataset, TransformDataset)
from chainer.links.model.vision import resnet
from chainercv import transforms

class ImageEvaluator(chainer.Chain):
    '''
    画像評価クラス image(sr, fi, sr + fi)
    '''
    def __init__(self, generator):
        super(ImageEvaluator, self).__init__()
        self.loss = None
        self.acc = None
        self.y = None

        with self.init_scope():
            self.generator = generator

    def __call__(self, x, t):
        self.y = None
        self.acc = None
        self.loss = None
        self.y = self.generator(x)
        self.loss = F.mean_squared_error(self.y, t)
        reporter.report({'loss':self.loss}, self)
        return self.loss


class SimpleFI(chainer.Chain):
    '''
    Simple Fi
    '''
    def __init__(self):
        super(SimpleFI, self).__init__()
        self.loss = None
        with self.init_scope():
            self.conv1 = L.Convolution2D(2, 32, ksize=3, stride=1, pad=1)
            self.conv2 = L.Convolution2D(32, 32, ksize=3, stride=1, pad=1)
            self.conv3 = L.Convolution2D(32, 1, ksize=3, stride=1, pad=1)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h
