'''
データセット
'''
from os import path
import random
import csv
import platform
import json

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

from tqdm import tqdm
import numpy as np
import chainer

#パス関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# STVSRのパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

# DATA_PATH = '/media/shimo/HDD_storage/DataSet'
DATA_PATH = path.join(ROOT_PATH, 'dataset')

# class SequenceDataset(chainer.dataset.DatasetMixin):
#     def __init__(self, dataset='train'):
#         self.image_paths = []
#         csv_path = None
#         if dataset == 'train':
#             csv_path = 'Train_Mini_UCF101/train_data_loc.csv'
#         elif dataset == 'test':
#             csv_path = 'Test_Mini_UCF101/train_data_loc.csv'

#         print("loading datasaet {} ...".format(dataset))
#         with open(path.join(DATA_PATH, csv_path)) as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 self.image_paths.append(path.join(DATA_PATH, row[0]))

#     def __len__(self):
#         return len(self.image_paths)

#     def get_example(self, i):
#         data = np.load(self.image_paths[i])
#         x_data = data['x_data']
#         y_data = data['y_data']
#         return transform_sequence(x_data, y_data)
#         # return x_data, y_data

# class SequenceDatasetOnMem(chainer.dataset.DatasetMixin):
#     def __init__(self, dataset='train'):
#         self.image_paths = []
#         csv_path = None
#         if dataset == 'train':
#             csv_path = 'Train_Mini_UCF101/train_data_loc.csv'
#         elif  dataset == 'test':
#             csv_path = 'Test_Mini_UCF101/train_data_loc.csv'

#         with open(path.join(DATA_PATH, csv_path)) as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 self.image_paths.append(path.join(DATA_PATH, row[0]))

#         data = np.load(self.image_paths[0])
#         nf, ch, h, w = data['x_data'].shape
#         self.x_data = np.zeros((len(self.image_paths), nf, ch, h, w), dtype=np.float32)
#         ch, h, w = data['y_data'].shape
#         self.y_data = np.zeros((len(self.image_paths), ch, h, w), dtype=np.float32)

#         print("loading dataset {} ...".format(dataset))
#         for i, p in tqdm(enumerate(self.image_paths)):
#             data = np.load(p)
#             self.x_data[i] = data['x_data']
#             self.y_data[i] = data['y_data']

#     def __len__(self):
#         return len(self.image_paths)

#     def get_example(self, i):
#         return transform_sequence(self.x_data[i], self.y_data[i])
#         # return self.x_data[i], self.y_data[i]

def transform_sequence(x_data, y_data):
    '''
    sequence データのデータAUGを行う
    '''
    # 水平方向に反転するかどうか
    if random.choice([True, False]):
        x_data = np.flip(x_data, 2)
        y_data = np.flip(y_data, 1)

    # 垂直方向に反転するかどうか
    if random.choice([True, False]):
        x_data = np.flip(x_data, 3)
        y_data = np.flip(y_data, 2)

    # 90度回転するかどうか
    if random.choice([True, False]):
        x_data = np.rot90(x_data, axes=(2, 3))
        y_data = np.rot90(y_data, axes=(1, 2))

    return x_data, y_data

class SequenceDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset='Test_Mini_UCF101'):
        self.image_paths = []
        csv_path = path.join(dataset, 'train_data_loc.csv')

        print("loading dataset {} ...".format(dataset))
        with open(path.join(DATA_PATH, csv_path)) as f:
            reader = csv.reader(f)
            for row in reader:
                self.image_paths.append(path.join(DATA_PATH, row[0]))

    def __len__(self):
        return len(self.image_paths)

    def get_example(self, i):
        data = np.load(self.image_paths[i])
        x_data = data['x_data']
        y_data = data['y_data']
        return transform_sequence(x_data, y_data)
        # return x_data, y_data

class SequenceDatasetOnMem(chainer.dataset.DatasetMixin):
    def __init__(self, dataset='Test_Mini_UCF101'):
        self.image_paths = []
        csv_path = path.join(dataset, 'train_data_loc.csv')

        with open(path.join(DATA_PATH, csv_path)) as f:
            reader = csv.reader(f)
            for row in reader:
                self.image_paths.append(path.join(DATA_PATH, row[0]))

        data = np.load(self.image_paths[0])
        nf, ch, h, w = data['x_data'].shape
        self.x_data = np.zeros((len(self.image_paths), nf, ch, h, w), dtype=np.float32)
        ch, h, w = data['y_data'].shape
        self.y_data = np.zeros((len(self.image_paths), ch, h, w), dtype=np.float32)

        print("loading dataset {} ...".format(dataset))
        for i, p in tqdm(enumerate(self.image_paths)):
            data = np.load(p)
            self.x_data[i] = data['x_data']
            self.y_data[i] = data['y_data']

    def __len__(self):
        return len(self.image_paths)

    def get_example(self, i):
        return transform_sequence(self.x_data[i], self.y_data[i])
        # return self.x_data[i], self.y_data[i]
