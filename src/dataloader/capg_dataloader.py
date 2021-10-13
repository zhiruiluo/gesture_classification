import logging

import pytorch_lightning as pl
import torch
from .capg_dataset import CapgDataset
from src.util.model_selection import DatasetSpliter
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger('dataloader.capg_dataloader')


def collate_fn(batch):
    X = []
    Label = []
    for data in batch:
        x, label = data
        X.append(x)
        Label.append(label)
    
    label = torch.stack(Label)
    logger.debug('[collate_fn] {}'.format(label))

    return X, label

class CapgDataLoader(pl.LightningDataModule):
    def __init__(self, dataname,winsize,stride,batch_size=32,nfold=10,shuffle=True,deterministic=True,num_workers=8,embedding=False):
        super().__init__()
        self.dataname = dataname
        self.winsize = winsize
        self.stride = stride
        self.batch_size=batch_size
        self.nfold = nfold
        self.shuffle = shuffle
        self.determ = deterministic
        self.num_workers = num_workers
        self.embedding = embedding

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
    
    def setup(self):
        logger.info('setup dataloader')
        dataset = CapgDataset(self.winsize, self.stride)
        self.dss = DatasetSpliter(dataset, dataset.key_name, strategy='tvt', nfold=self.nfold, deterministic=self.determ)
        self.dss.setup()
        self.nclass = dataset.nclass
        self.class_to_index = dataset.class_to_index
        self.input_shape = dataset.input_shape
        self.nfold = self.dss.nfold
        self.dataset = dataset
        logger.info('setup dataloader finished')

    def cv_gen(self) -> int:
        for i, data in enumerate(self.dss.split()):
            self._build_dataloader(*data)
            yield i
    
    def get_fold(self, fold):
        data = self.dss.get_fold(fold)
        self._build_dataloader(*data)

    def _to_dataloader(self, subset, shuffle, batch_size):
        dataloader = DataLoader(subset,
                batch_size=batch_size,
                pin_memory=True,
                drop_last=False,
                shuffle=shuffle,
                #collate_fn=collate_fn,
                num_workers=self.num_workers)
        return dataloader

    def _build_dataloader(self, train, val, test):
        dataloaders = {}
        dataloaders['train'] = self._to_dataloader(train, self.shuffle, self.batch_size)
        dataloaders['val'] = self._to_dataloader(val, False, self.batch_size)
        dataloaders['test'] = self._to_dataloader(test, False, self.batch_size)
        self.dataloaders = dataloaders

    
    def train_dataloader(self) -> DataLoader:
        return self.dataloaders['train']
    
    
    def val_dataloader(self) -> DataLoader:
        return self.dataloaders['val']
    
    
    def test_dataloader(self) -> DataLoader:
        return self.dataloaders['test']
