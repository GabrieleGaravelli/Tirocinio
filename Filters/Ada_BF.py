import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import serialize
from bloom import hashfunction as hashfunc
import os, sys;
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Filters import bloom as bl
from sklearn.base import BaseEstimator


class Ada_BloomFilter(BaseEstimator):
    def __init__(self, n, k_max):
        self.n = n
        self.h = []
        for i in range(int(k_max)):
            self.h.append(hashfunc(self.n))
        self.table = np.zeros(self.n, dtype=int)
    
    def fit(self, X, k, y=None):
        for i in range(int(k)):
            t = self.h[i](X)
            self.table[t] = 1
            
    def predict(self, X, k):
        is_element = True
        for i in range(int(k)):
            if self.v[self.h[i](X)] == 0:
                is_element = False
                break
        return is_element
    
    def score(self, X, k, y=None):
        y_hat = []
        for x in X:
            y_hat.append(self.predict(x,k))
        return np.sum(False != y_hat) / len(X)

class OptimalAdaBloomFilter(BaseEstimator):

    def __init__(self, opt_filter, opt_treshold, k_max):
        self.bloom_filter_opt = opt_filter
        self.thresholds_opt = opt_treshold
        self.k_max_opt = k_max
        
    def predict(self, X):
        query = X.iloc[:, 1]
        score = X.iloc[:, -1]
        test_result = []
        for score_s, query_s in zip(score, query):
            if score_s >= self.thresholds_opt[-2]:
                test_result.append(True)
            else:
                ix = min(np.where(score_s < self.thresholds_opt)[0])
                # thres = thresholds[ix]
                k = self.k_max_opt - ix
                test_result.append(self.bloom_filter_opt.predict(query_s, k))
        return test_result
    
    def score(self, X):     
        ML_positive = X.iloc[:, 1][(X.iloc[:, -1] >= self.thresholds_opt[-2])]
        query_negative = X.iloc[:, 1][(X.iloc[:, -1] < self.thresholds_opt[-2])]
        score_negative = X.iloc[:, -1][(X.iloc[:, -1] < self.thresholds_opt[-2])]
        test_result = []
        for score_s, query_s in zip(score_negative, query_negative):
            ix = min(np.where(score_s < self.thresholds_opt)[0])
            # thres = thresholds[ix]
            k = self.k_max_opt - ix
            test_result.append(self.bloom_filter_opt.predict(query_s, k))
        FP_items = sum(test_result) + len(ML_positive)
        return FP_items / len(X)
    
    def query(self, X):     
        ML_positive = X.iloc[:, 1][(X.iloc[:, -1] >= self.thresholds_opt[-2])]
        query_negative = X.iloc[:, 1][(X.iloc[:, -1] < self.thresholds_opt[-2])]
        score_negative = X.iloc[:, -1][(X.iloc[:, -1] < self.thresholds_opt[-2])]
        test_result = []
        for score_s, query_s in zip(score_negative, query_negative):
            ix = min(np.where(score_s < self.thresholds_opt)[0])
            # thres = thresholds[ix]
            k = self.k_max_opt - ix
            test_result.append(self.bloom_filter_opt.predict(query_s, k))
        FP_items = sum(test_result) + len(ML_positive)
        return FP_items

def R_size(count_key, count_nonkey, R0):
    R = [0]*len(count_key)
    R[0] = R0
    for k in range(1, len(count_key)):
        R[k] = max(int(count_key[k] * (np.log(count_nonkey[0]/count_nonkey[k])/np.log(0.618) + R[0]/count_key[0])), 1)
    return R

def train_opt_ADA(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = np.arange(c_min, c_max+10**(-6), 0.1)
    FP_opt = train_negative.shape[0]

    k_min = 0
    for k_max in range(num_group_min, num_group_max+1):
        for c in c_set:
            tau = sum(c ** np.arange(0, k_max - k_min + 1, 1))
            n = positive_sample.shape[0]
            hash_len = R_sum
            bloom_filter = Ada_BloomFilter(hash_len, k_max)
            thresholds = np.zeros(k_max - k_min + 1)
            thresholds[-1] = 1.1
            num_negative = sum(train_negative.iloc[:, -1] <= thresholds[-1])
            num_piece = int(num_negative / tau) + 1
            score = train_negative.iloc[:, -1][(train_negative.iloc[:, -1] <= thresholds[-1])]
            score = np.sort(score)
            for k in range(k_min, k_max):
                i = k - k_min
                score_1 = score[score < thresholds[-(i + 1)]]
                if int(num_piece * c ** i) < len(score_1):
                    thresholds[-(i + 2)] = score_1[-int(num_piece * c ** i)]

            query = positive_sample.iloc[:, 1]
            score = positive_sample.iloc[:, -1]

            for score_s, query_s in zip(score, query):
                ix = min(np.where(score_s < thresholds)[0])
                k = k_max - ix
                bloom_filter.fit(query_s, k)
            ML_positive = train_negative.iloc[:, 1][(train_negative.iloc[:, -1] >= thresholds[-2])]
            query_negative = train_negative.iloc[:, 1][(train_negative.iloc[:, -1] < thresholds[-2])]
            score_negative = train_negative.iloc[:, -1][(train_negative.iloc[:, -1] < thresholds[-2])]

            test_result = []
            
            for score_s, query_s in zip(score_negative, query_negative):
                ix = min(np.where(score_s < thresholds)[0])
                # thres = thresholds[ix]
                k = k_max - ix
                test_result.append(bloom_filter.predict(query_s, k))
            FP_items = sum(test_result) + len(ML_positive)
            

            if FP_opt > FP_items:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds
                k_max_opt = k_max
                print('False positive items: %d, Number of groups: %d, c = %f' %(FP_items, k_max, round(c, 2)))

    # print('Optimal FPs: %f, Optimal c: %f, Optimal num_group: %d' % (FP_opt, c_opt, k_max))
    return OptimalAdaBloomFilter(bloom_filter_opt, thresholds_opt, k_max_opt)

'''
Implement Ada-BF
'''
def main(DATA_PATH, R_sum, others):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_group_min', action="store", dest="min_group", type=int, required=True, help="Minimum number of groups")
    parser.add_argument('--num_group_max', action="store", dest="max_group", type=int, required=True, help="Maximum number of groups")
    parser.add_argument('--c_min', action="store", dest="c_min", type=float, required=True, help="minimum ratio of the keys")
    parser.add_argument('--c_max', action="store", dest="c_max", type=float, required=True, help="maximum ratio of the keys")

    results = parser.parse_args(others)
    num_group_min = results.min_group
    num_group_max = results.max_group
    c_min = results.c_min
    c_max = results.c_max
    '''
    Load the data and select training data
    '''
    data = serialize.load_dataset(DATA_PATH)
    train_negative = data.loc[(data['label'] == 0)]
    positive_sample = data.loc[(data['label'] == 1)]
    '''
    Plot the distribution of scores
    '''
    '''
    plt.style.use('seaborn-deep')
    x = data.loc[data['label']== 1,'score']
    y = data.loc[data['label']== 0,'score']
    bins = np.linspace(0, 1, 25)
    plt.hist([x, y], bins, log=True, label=['Keys', 'non-Keys'])
    plt.legend(loc='upper right')
    plt.savefig('./Score_Dist.png')
    plt.show()
    '''


    '''Stage 1: Find the hyper-parameters (spare 30% samples to find the parameters)'''
    opt_Ada = train_opt_ADA(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample)
    
    '''Stage 2: Run Ada-BF on all the samples'''
    ### Test Queries
    return opt_Ada

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")
    parser.add_argument('--size_of_Ada_BF', action="store", dest="R_sum", type=int, required=True, help="size of the Ada-BF")
    result =parser.parse_known_args() 
    main(result[0].data_path, result[0].R_sum, result[1])