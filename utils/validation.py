import numpy as np

from itertools import zip_longest

from collections import defaultdict

import copy

class Validator:
    ''' Random splitter. '''
    def __init__(self, ids):
        self.ids = ids
        self.n = len(ids)
        
    def split(self, n_splits, seed=19):
        r = np.random.RandomState(seed)
        permuted_ids = r.permutation(self.ids)
                                                                                                                                                                                                                                
        fold_size = self.n // n_splits
        assert fold_size, 'Not enough elements'
        folds = list(map(list, list(zip_longest(*[iter(permuted_ids)] * fold_size))))
              
        remainder = self.n - (fold_size * n_splits)
        
        if remainder:
            leftovers = folds.pop()[:self.n % fold_size] if self.n % fold_size else folds.pop()
            while len(leftovers) != remainder:
                leftovers += folds.pop()

            chosen_folds = r.choice(n_splits, size=remainder, replace=False)
            for idx, fold in zip(leftovers, chosen_folds):
                folds[fold].append(idx)
            
        splits = [
            (sum(folds[:i] + (folds[i + 1:] if i != n_splits - 1 else []), []), folds[i]) 
                for i in range(n_splits)
        ]
        
        return splits

class ClassValidator:
    ''' Multi-class splitter. '''
    def __init__(self, labels):
        class_ids = defaultdict(list)
        for idx, label in enumerate(labels):
            class_ids[label].append(idx)
        self.validators = [Validator(ids) for ids in class_ids.values()]
        
    def split(self, n_splits, seed=19):
        label_splits = [val.split(n_splits, seed) for val in self.validators]
        splits = list(map(lambda x: tuple(map(lambda y: sum(y, []), zip(*x))), zip(*label_splits)))
        return splits
    

class MultiClassValidator:
    ''' Hierarchical multi-class multi-dimensional splitter. '''
    def __init__(self, labels):
        self.class_ids = defaultdict(list)
        for idx, label in enumerate(map(tuple, labels)):
            self.class_ids[label].append(idx)
        self.dim = len(label)

    def split(self, n_splits, seed=19):
        
        class_ids = copy.deepcopy(self.class_ids)
        
        validators = []
        for i in range(self.dim):
            bad_ids = []
            bad_labels = []
            
            for label, ids in class_ids.items():
                if len(ids) // n_splits:
                    validators.append(Validator(ids))
                else:
                    bad_ids.append(ids)
                    bad_labels.append(label[:-1])
                    
            if not bad_labels or len(class_ids) == 1:
                break
                
            class_ids = defaultdict(list)
            for ids, label in zip(bad_ids, bad_labels):
                class_ids[label] += ids
        
        label_splits = [val.split(n_splits, seed) for val in validators]
        remainder = len(ids)
        if bad_labels and remainder >= n_splits:
            new_split = Validator(ids).split(n_splits, seed)
        else:
            new_split = Validator(ids).split(remainder, seed) + [(ids, [])] * (n_splits - remainder)

        label_splits.append(new_split)
        splits = list(map(lambda x: tuple(map(lambda y: sum(y, []), zip(*x))), zip(*label_splits)))
        
        return splits
