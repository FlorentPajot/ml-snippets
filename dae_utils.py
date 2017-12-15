import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SwapNoise(BaseEstimator, TransformerMixin):
    def __init__(self, ratio=.15, random_seed=123):
        self.seed = random_seed
        self.ratio = ratio
        
    def fit(self):
        return self
    
    def transform(self, input_data):
        
        x = np.zeros(np.shape(input_data))
        np.random.seed(self.seed)
        for c in range(np.shape(input_data)[1]):
            c_ = np.array(input_data)[:, c]
            x[:, c] = self.partial_transform(c_)
        return x

    def partial_transform(self, x):
        
        x_ = np.copy(x)
        swap_idx = np.where(np.random.rand(len(x)) < self.ratio)[0]
        np.put(x_, swap_idx, np.random.choice(x, len(swap_idx)))
        return x_        
        
    def fit_transform(self, input_data):
        self.fit()
        return self.transform(input_data)