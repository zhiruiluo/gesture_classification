import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict, Union, List
import logging

logger = logging.getLogger('dataset.base')

def encode_onehot(labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot, classes_dict

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        pass
    
    def __len__(self):
        raise NotImplemented

    def __getitem__(self, idx):
        raise NotImplemented

    def _find_class_(self, labels: Union[np.ndarray, List], one_hot: bool) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """convert class to indexes"""
        classes = sorted(np.unique(labels))
        self.class_to_index = {classname: i for i,
                                classname in enumerate(classes)}
        logger.info(f'class_to_index { self.class_to_index}')
        self.class_names = classes
        self.nclass = len(classes)
        self.indexes = [i for i in range(len(self.class_names))]
        index = np.vectorize(self.class_to_index.__getitem__)(labels)
        if one_hot:
            labels_onehot, classes_dict = encode_onehot(labels)
            self.classes_dict = classes_dict
            logger.info(f'classes_dict {self.classes_dict}')
            return index, labels_onehot
        
        return index, None