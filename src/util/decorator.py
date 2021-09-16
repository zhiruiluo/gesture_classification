import pickle
import os
import pandas as pd
import joblib
from typing import Any
from time import time
import logging

logger = logging.getLogger('util.decorator')

__all__ = ['buffer_value']

def pandas_to_file(df: pd.DataFrame, path: str):
    df.to_csv(path)

def pandas_from_file(path: str):
    return pd.read_csv(path)

def pickle_to_file(object: Any, path: str):
    pickle.dump(object,open(path,'wb'))

def pickle_from_file(path: str):
    return pickle.load(open(path,'rb'))

def joblib_to_file(object: Any, path: str, compress: int = 0):
    joblib.dump(object, path, compress=compress)

def joblib_from_file(path: str):
    return joblib.load(path)

def protocol_writer(protocol):
    if protocol == 'pandas':
        return pandas_to_file
    elif protocol == 'pickle':
        return pickle_to_file
    elif protocol == 'joblib':
        return joblib_to_file
    else:
        raise ValueError(f'buffer_value protocol_writer Error: get {protocol}!')

def protocol_reader(protocol):
    if protocol == 'pandas':
        return pandas_from_file
    elif protocol == 'pickle':
        return pickle_from_file
    elif protocol == 'joblib':
        return joblib_from_file
    else:
        raise ValueError(f'buffer_value protocol_reader Error: get {protocol}!')
    
def protocol_postfix(protocol):
    if protocol == 'pandas':
        return '.csv'
    elif protocol == 'pickle':
        return '.pickle'
    elif protocol == 'joblib':
        return '.pkl'
    else:
        raise ValueError(f'buffer_value protocol_postfix Error: get {protocol}!')

def buffer_value(protocol, folder):
    '''decorator for buffering temporary values in files\n
    protocol: [ 'pandas' | 'pickle' | 'joblib' ]\n
    folder: user defined path
    '''
    def decorator(func):
        def BufferWrapper(fn, *args, **kwargs):
            if not os.path.isdir(folder):
                os.mkdir(folder)
            
            fpath = os.path.join(os.path.join(folder,fn+protocol_postfix(protocol)))
            if not os.path.isfile(fpath):
                out = func(*args,**kwargs)
                writer = protocol_writer(protocol)
                t1 = time()
                writer(out, fpath)
                logger.info(f'Writer object to {fpath}, @{protocol}, spent time {time()-t1}')
            else:
                reader = protocol_reader(protocol)
                t1 = time()
                out = reader(fpath)
                logger.info(f'Read object from {fpath}, @{protocol}, spent time {time()-t1}')
            return out
        return BufferWrapper
    return decorator
