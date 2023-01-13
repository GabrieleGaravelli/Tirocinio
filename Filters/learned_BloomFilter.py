from turtle import pos
import numpy as np
import pandas as pd
import argparse
import serialize
import os, sys;
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Filters import bloom as bl
from sklearn.base import BaseEstimator

class LBF(BaseEstimator):
    def __init__(self, n, threshold):
        self.n = n
        self.threshold = threshold
        self.fitted = False
        
        
    def fit(self, X, y=None):
        '''Initialize the bit array with positive elements
        
        - X (array of int): array containing all the positive elements
        - y (array of boolean): array containing labels for X's elements
        '''
        self.keys = X[(X.iloc[:, -1] <= self.threshold)]
        self.k = max(1,int(self.n/len(X)*np.log(2)))
        self.backup_bf = bl.BloomFilter(self.n, self.k)
        self.backup_bf.fit(self.keys.iloc[:, 1])
        
        self.fitted = True
        
    def predict(self, X):
        '''Return which elements are positive and which not
        
        - X (array of int): elements to filter
        
        Returns: 
            an array of boolean indicating filter results 
        '''
        if not self.fitted:
            raise ValueError('BloomFilter object not fitted')
   
        result = []
        query_set = X.iloc[:, 1]
        score = X.iloc[:, -1]

        for el, sc in zip(query_set, score):
            is_element = True
            if sc <= self.threshold:
                is_element = self.backup_bf.predict([el])
            result.append(is_element)
        return np.array(result)
        
    def score(self, X, y=None):
        '''Calculate an empirical false positive rate
        
        - X (array of int): array containing only non-positive elements
        - y (array of boolean): array containing labels for X's elements
        
        Returns:
            a float false positive rate
        '''
        if not self.fitted:
                raise ValueError('BloomFilter object not fitted')
            
        th_positive = X.iloc[:, 0][(X.iloc[:, 1] > self.threshold)]
        th_negative = X.iloc[:, 0][(X.iloc[:, 1] <= self.threshold)]
        bf_positive = th_negative[self.backup_bf.predict(th_negative)]
        n_fp = len(th_positive) + len(bf_positive)
        return n_fp / len(X)

    def query(self, query_set):
        '''
        Test the LBF against the negative queries in input. 
        Returns the number of false positives obtained, i.e - the number of negative queries classified as positive by the filter
        query_set: df in the following form
            (index)     data    label    score
        '''

        ml_positive = query_set.iloc[:, 1][(query_set.iloc[:, -1] > self.threshold)]
        bloom_negative = query_set.iloc[:, 1][(query_set.iloc[:, -1] <= self.threshold)]
        bf_positive = self.backup_bf.test(bloom_negative, single_key = False)
        fp_items = sum(bf_positive) + len(ml_positive)

        return fp_items

def train_lbf(filter_size, query_train_set, keys, quantile_order):
    '''Search for the best threshold'''
    train_dataset = np.array(pd.concat([query_train_set, keys]).iloc[:, -1])
    thresholds_list = [np.quantile(train_dataset, i * (1 / quantile_order)) for i in range(1, quantile_order)] if quantile_order < len(train_dataset) else np.sort(train_dataset)
    # thresh_third_quart_idx = (3 * len(thresholds_list) - 1) // 4

    fp_opt = query_train_set.shape[0]
    lbf_opt = LBF(filter_size, 1.0) # caso base
    lbf_opt.fit(keys)

    for threshold in thresholds_list:
        lbf = LBF(filter_size, threshold)
        lbf.fit(keys)
        fp_items = lbf.query(query_train_set)
        
        # Se con la soglia provata miglioro il numero di falsi positivi aggiorno la soglia corrente
        if fp_items < fp_opt:
            fp_opt = fp_items
            lbf_opt = lbf
            print(f"Current threshold: {threshold}, False positive items: {fp_items}")
    
    print(f"Chosen thresholds: {lbf_opt.threshold}")

    return lbf_opt, fp_opt

def main(data_path, size_filter, others):
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresholds_q', action = "store", dest = "thresholds_q", type = int, required = True, help = "order of quantiles to be tested")
    results = parser.parse_args(others)

    thresholds_q = results.thresholds_q
    DATA_PATH = data_path
    R_sum = size_filter

    '''
    Load the data and select training data.
    '''
    data = serialize.load_dataset(DATA_PATH)
    train_negative = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]
    
    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    lbf_opt, _ = train_lbf(R_sum, train_negative, positive_sample, thresholds_q)
    
    return lbf_opt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")
    result =parser.parse_known_args() 
    print(result[0])
    main(result[0].data_path, result[1])
