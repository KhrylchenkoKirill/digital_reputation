import numpy as np
import pandas as pd

from collections import Counter

class Tokenizer:
    ''' Tokenizer for mapping token values to indices and replacing rare tokens with OOV token. '''
    def __init__(self, min_df):
        self.min_df = min_df
        self.vocab = None
        
    def transform(self, df):
        if self.vocab is None:
            cnt = Counter(df['A'])
            self.vocab = ['OOV'] + [el for el, freq in cnt.most_common() if freq >= self.min_df]
            self.vocab_size = len(self.vocab) + 1
            self.token2idx = dict(zip(self.vocab, range(self.vocab_size)))
        df['num'] = df['A'].map(lambda x: self.token2idx.get(x, 0))
        corpus = df.groupby('id')['num'].apply(list)
        return corpus
