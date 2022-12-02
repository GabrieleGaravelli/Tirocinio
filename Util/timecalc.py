import numpy as np
import timeit
import os, sys; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Filters import bloom as blb
from Filters import bloom_int as bli


x1 = '''
def bloomint2():
    X = [7800, 4635, 67, 903, 12, 4, 921, 3030, 22, 7010]
    no_X = np.random.uniform(low=0, high=8000, size=1000)
    bf = bli.BloomFilter(100, [bli.h1, bli.h2, bli.h3])
    bf.fit(X)
    assert (bf.predict(X) == True).all()
    s = set(no_X)
    for x in X:
        if x in s:
            s.remove(x)
    no_X = np.array(list(s)) 
    emp_fpr = bf.score(no_X)
    theo_fpr = bf.false_positive_prob()
    '''
        
x2 = '''
def bloombit2():
    X = [7800, 4635, 67, 903, 12, 4, 921, 3030, 22, 7010]
    no_X = np.random.uniform(low=0, high=8000, size=1000)
    bf = blb.BloomFilter(100, [blb.h1, blb.h2, blb.h3])
    bf.fit(X)
    assert (bf.predict(X) == True).all()
    s = set(no_X)
    for x in X:
        if x in s:
            s.remove(x)
    no_X = np.array(list(s)) 
    emp_fpr = bf.score(no_X)
    theo_fpr = bf.false_positive_prob()
    '''
        
n = 100
resultint = timeit.timeit(stmt=x1, number=n)
resultbit = timeit.timeit(stmt=x2, number=n)
    
print(f'Int Time: {resultint}')
print(f'Bit Time: {resultbit}')


    

    
