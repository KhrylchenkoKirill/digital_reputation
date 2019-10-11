import numpy as np
import pandas as pd

from utils.validation import ClassValidator

from sklearn.metrics import roc_auc_score

class CVModel:
    ''' Framework for the cross-validation of a given model. Enables custom validation splitting support. '''
    def __init__(self, get_model, fit_model):
        self.get_model = get_model
        self.fit_model = fit_model
        self.models = None
        
    def fit(self, X, y, n_splits, seed=19, validator=None, cache=False, sparse=False, verbose=False):
        
        val = ClassValidator(y) if validator is None else validator
        
        scores = []
        self.models = []
        if cache:
            val_preds = np.zeros(X.shape[0])
        for fold, (train_ids, val_ids) in enumerate(val.split(n_splits, seed)):
            
            if sparse:
                train_X, test_X = X[train_ids], X[val_ids]
            else:
                train_X, test_X = X.iloc[train_ids], X.iloc[val_ids]

            train_y, test_y = y.iloc[train_ids], y.iloc[val_ids]
            model = self.get_model()
            self.models.append(model)
            score = self.fit_model(model, train_X, train_y, test_X, test_y)
            scores.append(score)
            
            if verbose:
                print('fold: {:>2}, score: {:>4f}'.format(fold + 1, score))
            if cache:
                val_preds[val_ids] = model.predict_proba(test_X)[:, 1]
            
        if cache:
            return np.mean(scores), val_preds
        else:
            return np.mean(scores)
        
    def predict(self, X):
        return np.mean([model.predict_proba(X)[:, 1] for model in self.models], axis=0)
