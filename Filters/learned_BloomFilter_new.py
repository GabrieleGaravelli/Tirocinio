import numpy as np
import bloom as bl
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_consistent_length

class LBF(BaseEstimator):
        def __init__(self, nhash, n, threshold):
            self.n = n
            self.threshold = threshold
            self.backup_bf = bl.BloomFilter(self.n, nhash)
            self.fitted = False
            
        def fit(self, X, y=None):
            
            if y is not None:
                check_consistent_length(X, y)
                
            if y is not None and (np.array(y) == False).any():
                raise ValueError('y in fit cannot contain negative labels')
            
            score = X.iloc[:, 1]

            self.keys = X.iloc[:,0][(score <= self.threshold)]
            self.backup_bf.fit(self.keys)
            
            self.fitted = True
            
            return self
        
        def predict(self, X):
            
            if not self.fitted:
                raise ValueError('BloomFilter object not fitted')
   
            result = []
            query_set = X.iloc[:, 0]
            score = X.iloc[:, 1]
            
            for el, sc in zip(query_set, score):
                is_element = True
                if sc <= self.threshold:
                    is_element = self.backup_bf.predict([el])
                result.append(is_element)
            return np.array(result)
        
        def score(self, X, y=None):
            
            if y is not None:
                check_consistent_length(X, y)
            
            if y is not None and (np.array(y) == True).any():
                raise ValueError('y in score cannot contain positive labels')

            if not self.fitted:
                raise ValueError('BloomFilter object not fitted')
            
            th_positive = X.iloc[:, 0][(X.iloc[:, 1] > self.threshold)]
            th_negative = X.iloc[:, 0][(X.iloc[:, 1] <= self.threshold)]
            bf_positive = th_negative[self.backup_bf.predict(th_negative)]
            n_fp = len(th_positive) + len(bf_positive)
            return n_fp
        
        @staticmethod
        def train_lbf(n, neg_query_set, keys, quantile_order):
            threshold_list =  range(quantile_order)#non ho capito bene il criterio
            nhash = 7
            
            fp_opt = neg_query_set.shape[0]
            lbf_opt = LBF(nhash, n, 1)
            
            for th in threshold_list:
                lbf = LBF(nhash, n, th)
                lbf.fit(keys)
                fp_act = lbf.score(neg_query_set)
                
                if fp_act < fp_opt:
                    fp_opt = fp_act
                    lbf_opt = lbf
            return lbf_opt, fp_opt

def main():
    
    n = 10001
    max_th = 500
    
    
    X = np.random.randint(low=0, high=800000, size=1000)
    X_score = np.random.randint(low=0, high=max_th, size=1000) 
    X = pd.DataFrame({'data':X, 'score':X_score})
    
    no_X = np.random.randint(low=0, high=800000, size=100000) 
    s = set(no_X) 
    for x in X.iloc[:, 0]:
        if x in s:
            s.remove(x)
    no_X_score = np.random.randint(low=0, high=max_th, size=len(s))
    no_X = pd.DataFrame({'data':np.array(list(s)), 'score':no_X_score})
    
    lbf, fp = LBF.train_lbf(n, no_X, X, max_th)
    print(f'{lbf.threshold} {fp} {len(s)}')
    
if __name__ == '__main__':
    main()