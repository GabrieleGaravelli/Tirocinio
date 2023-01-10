import numpy as np
import os, sys; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Filters import bloom as bl


def main():

    bf = bl.BloomFilter(n = 100, nhash = 3, fp_rate = None)
    
    X = [7800, 4635, 67, 903, 12, 4, 921, 3030, 22, 7010]
    
    bf.fit(X)
    
    assert (bf.predict(X) == True).all()

    no_X = np.random.randint(low=0, high=8000, size=1000)
    s = set(no_X)
    for x in X:
        if x in s:
            s.remove(x)
    no_X = np.array(list(s))
    
    emp_fpr = bf.score(no_X)
    theo_fpr = bf.false_positive_prob()

    print(f'empirical FPR = {emp_fpr:.3f}') 
    print(f'theorical FPR = {theo_fpr:.3f}')

    #assert abs(emp_fpr - theo_fpr) < 1E-2

if __name__ == '__main__':
    main()