import math
import numpy as np
import bitarray as ba
import random
from sklearn.utils import murmurhash3_32

from sklearn.base import BaseEstimator
from sklearn.utils import check_consistent_length


class BloomFilter(BaseEstimator):

    def __init__(self, n, hash):
        '''Initialize a Bloom Filter.

        - n (int): dimension of the array backing the Bloom Filter
        - hash (list of function): hash functions backing the Bloom Filter
        '''

        self.n = n
        self.hash = hash
        self.v = ba.bitarray(n) 
        self.v.setall(0)
        self.fitted = False

    def fit(self, X, y=None):
        '''Initialize the bit array with positive elements
        
        - X (array of int): array containing all the positive elements
        - y (array of boolean): array containing labels for X's elements
        '''
        
        if y is not None:
            check_consistent_length(X, y)
            
        if y is not None and (np.array(y) == False).any():
            raise ValueError('y in fit cannot contain negative labels')

        self.m = len(X)

        for x in X:
            for h in self.hash:
                self.v[h(x)] = 1

        self.fitted = True
        
        return self

    def predict(self, X):
        '''Return which elements are positive and which not
        
        - X (array of int): elements to filter
        
        Returns: 
            an array of boolean indicating filter results 
        '''
        
        if not self.fitted:
            raise ValueError('BloomFilter object not fitted')
   
        result = []

        for x in X:
            is_element = True
            for h in self.hash:
                if self.v[h(x)] == 0:
                    is_element = False
                    break
            result.append(is_element)
        return np.array(result)

    def score(self, X, y=None):
        '''Calculate an empirical false positive rate
        
        - X (array of int): array containing only non-positive elements
        - y (array of boolean): array containing labels for X's elements
        
        Returns:
            a float false positive rate
        '''
        if y is not None:
            check_consistent_length(X, y)
            
        if y is not None and (np.array(y) == True).any():
            raise ValueError('y in score cannot contain positive labels')

        if not self.fitted:
            raise ValueError('BloomFilter object not fitted')
   
        y_hat = self.predict(X)
        return np.sum(False != y_hat) / len(X)

    def false_positive_prob(self):
        '''Returns the theorical false positive rate
        
        Returns:
            a float false positive rate
        '''
        if not self.fitted:
            raise ValueError('BloomFilter object not fitted')

        k = len(self.hash)
        n = len(self.v)
        m = self.m
        return (1 - math.e ** (-k * m / n)) ** k
 
class hashfunction(object):
    def __init__(self, m):
        self.m = m
        self.ss = random.randint(1,99999999)
    def __call__(self, x):
        return murmurhash3_32(x, seed = self.ss) % self.m

def h1(x, n):
    '''Simple hash function
    
    - x (int): number to hash
    - n (int): size of the hash table
    '''
    return int(x % n)

def h2(x, n):
    '''Simple hash function
    
    - x (int): number to hash
    - n (int): size of the hash table
    '''
    return math.floor(n * (x * 0.77 % 1))

def h3(x, n):
    '''Simple hash function
    
    - x (int): number to hash
    - n (int): size of the hash table
    '''
    return int(hash(str(x)) % n)










