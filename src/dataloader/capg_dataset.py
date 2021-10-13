import logging
import os
import re
from random import shuffle

import pandas as pd
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from src.util.decorator import buffer_value

from .base_dataset import BaseDataset

logger = logging.getLogger('dataset.CapgDataset')

@buffer_value('joblib','temporary')
def load_mat(data_path, normalization=True):
    labels = []
    data = []
    for fn in sorted(os.listdir(data_path)):
        if fn.endswith('.mat'):
            pth = os.path.join(data_path, fn)
            datum = loadmat(pth)
            label = re.split('\.|-', fn)[0:3]
            # subject, gesture, trail
            data.append(datum['data'])
            labels.append(label[1])
    if normalization:
        data = list(normalize(data))
    
    data, labels = shuffle_dataset(data,labels)

    return data, labels

def normalize(data):
    ss = StandardScaler()
    data = np.array(data)
    n, t, v = data.shape
    data = data.reshape(v, t*n)
    #logger.info(f'reshape {data.shape}')
    data = np.array([ss.fit_transform(np.array(d).reshape(-1,1)).reshape(-1) for d in data])
    #logger.info(f'fit {data.shape}')
    data = data.reshape(n, t, v)
    #logger.info(f'reshape {data.shape}')
    return data

def shuffle_dataset(data, labels):
    index_shuf = list(range(len(data)))
    data_shuf = []
    labels_shuf = []
    shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        labels_shuf.append(labels[i])

    return data_shuf, labels_shuf

def slide_window_tensor(tensor: torch.Tensor, dim, win_size, stride):
    n_tensor = tensor.unfold(dim,win_size,stride).permute(0, 2, 1)
    return [t for t in n_tensor] 

class CapgDataset(BaseDataset):
    def __init__(self, winsize, stride) -> None:
        super().__init__()
        self.data_path = '../data/'
        self.winsize = winsize
        self.stride = stride
        self._load()

    @property
    def key_name(self):
        return f'CapgDataset_{self.winsize}_{self.stride}'

    def _load(self):
        self.data, self.labels = load_mat('capgdataset',data_path=self.data_path)
        self.labels, _ = self._find_class_(self.labels, False)

        self.input_shape = self.data[0].shape
        self._to_tensor()
        self._seperate_by_trial()

    def _seperate_by_trial(self):
        new_data = []
        new_label = []
        for d, l in zip(self.data,self.labels):
            window_data = slide_window_tensor(d, 0, self.winsize, self.stride)
            new_data.extend(window_data)
            new_label.extend([l]*len(window_data))
        self.data = new_data
        self.labels = new_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
         
    def _to_tensor(self):
        self.data = [torch.tensor(d, dtype=torch.float32) for d in self.data]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in self.labels]
