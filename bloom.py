import math
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y


class BloomFilter(BaseEstimator):

    def __init__(self, n, hash):
        '''Initialize a Bloom Filter.

        - n (int): dimension of the array backing the Bloom Filter
        - hash (list of function): hash functions backing the Bloom Filter
        '''

        self.n = n
        self.hash = hash
        self.v = [0] * n
        self.fitted = False

    def fit(self, X, y=None):
        if y is not None and (np.array(y) == False).any():
            raise ValueError('y in fit cannot contain negative labels')

        if y is not None:
            check_X_y(X, y)

        self.m = len(X)

        for x in X:
            for h in self.hash:
                self.v[h(x)] = 1

        self.fitted = True
        
        return self

    def predict(self, X):
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
        if y is not None and (np.array(y) == True).any():
            raise ValueError('y in score cannot contain positive labels')
        
        if y is not None:
            check_X_y(X, y)

        if not self.fitted:
            raise ValueError('BloomFilter object not fitted')
   
        y_hat = self.predict(X)
        return np.sum(False != y_hat) / len(X)

    def false_positive_prob(self):
        if not self.fitted:
            raise ValueError('BloomFilter object not fitted')

        k = len(self.hash)
        n = len(self.v)
        m = self.m
        return (1 - math.e ** (-k * m / n)) ** k
            
        
    # def filter(self, T):
    #     self.result = set()
    #     for x in T:
    #         if self.n[h1(x, self.ts)] == 1 and self.n[h2(x, self.ts)] == 1 and self.n[h3(x, self.ts)] == 1:
    #             self.result.add(x)
    #     fp = (1 - math.e**((-self.k)*len(self.S)/self.ts))**self.k
    #     print(self.result.issuperset(self.S))
    #     print('Theoretical false positive rate: {}'.format(fp))
    #     print('Actual false positive rate: {}'.format((len(self.result) - len(self.S))/(len(T) - len(self.S))))
    #     return self.result 

n = 100

def h1(x):
    return int(x % n)

def h2(x):
    return math.floor(n * (x * 0.77 % 1))

def h3(x):
    return int(hash(str(x)) % n)

if __name__ == '__main__':
    bf = BloomFilter(n, [h1, h2, h3])

    X = [7800, 4635, 67, 903, 12, 4, 921, 3030, 22, 7010]
    bf.fit(X)

    assert (bf.predict(X) == True).all()

    no_X = np.random.uniform(low=0, high=8000, size=1000)
    s = set(no_X)
    for x in X:
        if x in s:
            s.remove(x)
    no_X = np.array(list(s))

    emp_fpr = bf.score(no_X)
    theo_fpr = bf.false_positive_prob()

    print(f'empirical FPR = {emp_fpr:.3f}')
    print(f'theorical FPR = {theo_fpr:.3f}')

    assert abs(emp_fpr - theo_fpr) < 1E-2








