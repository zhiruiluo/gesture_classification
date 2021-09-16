import os
from typing import List, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from operator import itemgetter
from torch.utils.data import Subset

import logging
logger = logging.getLogger('utils.model_selection')

__all__ = ['DataSpliter']


class DataSpliter():
    def __init__(self, X, y, prefix, strategy, cv=False, nfold=5, deterministic=True):
        self.X = X
        self.y = y
        self.prefix = prefix
        self.strategy = strategy
        self.cv = cv
        self.nfold = nfold
        self.deterministic = deterministic
        self.kfold_path = './kfold/'
    
    def get_fold(self, fold) -> Tuple[List, List, List]:
        if self.kfolds is None:
            raise UnboundLocalError('Please set kfold deterministic to True')
        
        df_fold = self.kfolds[fold]
        train, val, test = self._index_to_split(df_fold, self.X, self.y)
        
        logger.info(f'[kfold summary] train {len(train)} val {len(val)} test {len(test)}')

        return train, val, test

    def split(self) -> Tuple[List, List, List]:
        if self.kf is None:
            raise UnboundLocalError('When kfold deterministic is true, please call kfold_get_by_fold')
        for fold, idx in enumerate(self._split_strategy(self.y, self.strategy), 0):
            df_fold = self._combine_split(*idx)
            train, val, test = self._index_to_split(df_fold, self.X, self.y)
            yield train, val, test

    def setup(self):
        if self.deterministic:
            logger.info('setup Dataspliter determinstic')
            self.kfolds = self._kfold_index_build(self.X, self.y, self.nfold, self.strategy, self.deterministic)
        else:
            logger.info('setup Dataspliter StratifiedKFold')
            self.kf = StratifiedKFold(n_splits=self.nfold,shuffle=True)
            
    def _kfold_index_build(self, X, y, nfold, strategy, deterministic):
        if not os.path.isdir(self.kfold_path):
            os.mkdir(self.kfold_path)
        
        fold_generated = True
        kfolds = {}
        for fold in range(nfold):
            path = os.path.join(self.kfold_path,f'{self.prefix}_{strategy}_nfold={self.nfold}_{fold}.csv')
            if not os.path.isfile(path):
                fold_generated = False
                break
            df_fold = pd.read_csv(path)
            kfolds[fold] = df_fold
        
        if fold_generated:
            return kfolds

        kfolds = {}
        for fold, idx in enumerate(self._split_strategy(y, strategy), 0):
            df_fold = self._combine_split(*idx)
            path = os.path.join(self.kfold_path,f'{self.prefix}_{strategy}_nfold={self.nfold}_{fold}.csv')
            df_fold.to_csv(path, index=False)
            kfolds[fold] = df_fold
        
        return kfolds

    def _combine_split(self, train_idx, val_idx, test_idx) -> pd.DataFrame:
        train_idx = [(i, 0) for i in train_idx]
        test_idx = [(i, 2) for i in test_idx]

        df_train = pd.DataFrame(train_idx, columns=['index','train_type'])
        df_test = pd.DataFrame(test_idx, columns=['index','train_type'])
        if val_idx is not None:
            val_idx = [(i,1) for i in val_idx]
            df_val = pd.DataFrame(val_idx, columns=['index','train_type'])
            df_tuple = (df_train, df_val, df_test)
        else:
            df_tuple = (df_train, df_test)
        
        df_fold = pd.concat(df_tuple, axis=0, ignore_index=True)
        return df_fold

    def _index_to_split(self, df_fold, X, y) -> Tuple[List, List, List]:
        train_index = df_fold[df_fold['train_type']==0]['index'].to_numpy()
        val_index = df_fold[df_fold['train_type']==1]['index'].to_numpy()
        test_index = df_fold[df_fold['train_type']==2]['index'].to_numpy()

        if val_index.shape[0] == 0:
            val_index = test_index
        
        if X is not None:
            X_train, X_val, X_test = itemgetter(*train_index)(X), itemgetter(*val_index)(X), itemgetter(*test_index)(X)
        y_train, y_val, y_test = itemgetter(*train_index)(y), itemgetter(*val_index)(y), itemgetter(*test_index)(y)
        train = [[data, label] for data, label in zip(X_train, y_train)]
        val = [[data, label] for data, label in zip(X_val, y_val)]
        test = [[data, label] for data, label in zip(X_test, y_test)]

        return train, val, test

    def _split_strategy(self, y, strategy):
        kf = StratifiedKFold(n_splits=self.nfold)
        for train_index, test_index in kf.split(np.zeros(len(y)), y):
            val_index = None
            if strategy == 'tvt':
                y_train = y[train_index]
                idx_train, idx_val = train_test_split(
                    train_index, train_size=0.8, random_state=1, shuffle=True, stratify=y_train)

                train_index = idx_train
                val_index = idx_val
            elif strategy == 'ttv':
                y_test = y[test_index]
                idx_val, idx_test = train_test_split(
                    test_index, train_size=0.5, random_state=1, shuffle=True, stratify=y_test)

                test_index = idx_test
                val_index = idx_val
            
            yield train_index, val_index, test_index


class DatasetSpliter():
    def __init__(self, dataset, prefix, strategy, nfold=5, deterministic=True):
        self.dataset = dataset
        self.prefix = prefix
        self.strategy = strategy
        self.nfold = nfold
        self.deterministic = deterministic
        self.kfold_path = './kfold/'
    
    def get_fold(self, fold) -> Tuple[List, List, List]:
        if self.kfolds is None:
            raise UnboundLocalError('Please set kfold deterministic to True')
        
        df_fold = self.kfolds[fold]
        train, val, test = self._index_to_subset(df_fold, self.dataset)
        
        logger.info(f'[kfold summary] train {len(train)} val {len(val)} test {len(test)}')

        return train, val, test

    def split(self) -> Tuple[List, List, List]:
        if self.kf is None:
            raise UnboundLocalError('When kfold deterministic is true, please call kfold_get_by_fold')
        for fold, idx in enumerate(self._split_strategy(self.dataset.labels, self.strategy), 0):
            df_fold = self._combine_split(*idx)
            train, val, test = self._index_to_subset(df_fold, self.dataset)
            yield train, val, test

    def setup(self):
        if self.deterministic:
            logger.info('setup Dataspliter determinstic')
            self.kfolds = self._kfold_index_build(self.dataset, self.nfold, self.strategy)
        else:
            logger.info('setup Dataspliter StratifiedKFold')
            self.kf = StratifiedKFold(n_splits=self.nfold,shuffle=True)
    
    def _count_by_class(self, phase, dataset):
        count = defaultdict(int)

        for data in dataset:
            count[data[-1].item()] += 1
        logger.info('[{} class] {}'.format(phase, [ (k,v) for k,v in sorted(count.items(), key=lambda item: item[0])]))

    def _kfold_index_build(self, dataset, nfold, strategy):
        if not os.path.isdir(self.kfold_path):
            os.mkdir(self.kfold_path)
        
        fold_generated = True
        kfolds = {}
        for fold in range(nfold):
            path = os.path.join(self.kfold_path,f'{self.prefix}_{strategy}_nfold={self.nfold}_{fold}.csv')
            if not os.path.isfile(path):
                fold_generated = False
                break
            df_fold = pd.read_csv(path)
            kfolds[fold] = df_fold
        
        if fold_generated:
            return kfolds

        kfolds = {}
        for fold, idx in enumerate(self._split_strategy(dataset.labels, strategy), 0):
            df_fold = self._combine_split(*idx)
            path = os.path.join(self.kfold_path,f'{self.prefix}_{strategy}_nfold={self.nfold}_{fold}.csv')
            df_fold.to_csv(path, index=False)
            kfolds[fold] = df_fold
        
        return kfolds

    def _combine_split(self, train_idx, val_idx, test_idx) -> pd.DataFrame:
        train_idx = [(i, 0) for i in train_idx]
        test_idx = [(i, 2) for i in test_idx]

        df_train = pd.DataFrame(train_idx, columns=['index','train_type'])
        df_test = pd.DataFrame(test_idx, columns=['index','train_type'])
        if val_idx is not None:
            val_idx = [(i,1) for i in val_idx]
            df_val = pd.DataFrame(val_idx, columns=['index','train_type'])
            df_tuple = (df_train, df_val, df_test)
        else:
            df_tuple = (df_train, df_test)
        
        df_fold = pd.concat(df_tuple, axis=0, ignore_index=True)
        return df_fold

    def _index_to_subset(self, df_fold, dataset) -> Tuple[List, List, List]:
        train_index = df_fold[df_fold['train_type']==0]['index'].to_numpy()
        val_index = df_fold[df_fold['train_type']==1]['index'].to_numpy()
        test_index = df_fold[df_fold['train_type']==2]['index'].to_numpy()

        if val_index.shape[0] == 0:
            val_index = test_index

        train = Subset(dataset, train_index)
        val = Subset(dataset, val_index)
        test = Subset(dataset, test_index)
        self._count_by_class('train',train)
        self._count_by_class('val',val)
        self._count_by_class('test',test)

        return train, val, test

    def _split_strategy(self, y, strategy):
        kf = StratifiedKFold(n_splits=self.nfold)
        for train_index, test_index in kf.split(np.zeros(len(y)), y):
            val_index = None
            if strategy == 'tvt':
                y_train = itemgetter(*train_index)(y)
                idx_train, idx_val = train_test_split(
                    train_index, train_size=0.8, random_state=1, shuffle=True, stratify=y_train)

                train_index = idx_train
                val_index = idx_val
            elif strategy == 'ttv':
                y_test = itemgetter(*test_index)(y)
                idx_val, idx_test = train_test_split(
                    test_index, train_size=0.5, random_state=1, shuffle=True, stratify=y_test)

                test_index = idx_test
                val_index = idx_val
            
            yield train_index, val_index, test_index
