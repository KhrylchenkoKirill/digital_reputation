import numpy as np
import pandas as pd

import os

filenames = ['X1', 'X2', 'X3', 'Y']

class Data:
    ''' Training data loader. '''
    def __init__(self, path):
        for name in filenames:
            self.__dict__[name] = pd.read_csv(os.path.join(path, '{}.csv'.format(name)), index_col='id')
            
test_filenames = ['X1', 'X2', 'X3']

class TestData:
    ''' Test data loader. '''
    def __init__(self, path):
        for name in test_filenames:
            self.__dict__[name] = pd.read_csv(os.path.join(path, '{}.csv'.format(name)), index_col='id')
