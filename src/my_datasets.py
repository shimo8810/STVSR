from os import path
import h5py
import numpy as np
import chainer
from chainer import datasets

# Path config
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

class SequenceDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset='SCENE1'):
        if dataset == 'SCENE10':
            h5file = h5py.File(path.join(ROOT_PATH, 'dataset/SCENE10_2K_60fps.hdf5'))
        elif dataset == 'SCENE1':
            h5file = h5py.File(path.join(ROOT_PATH, 'dataset/SCENE1_2K_for_fi.hdf5'))
        self.x_data = h5file['x_data']
        self.y_data = h5file['y_data']

    def __len__(self):
        return len(self.x_data)

    def get_example(self, i):
        return self.x_data[i], self.y_data[i]
