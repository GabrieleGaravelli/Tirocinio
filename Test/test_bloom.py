import unittest
import numpy as np
import os, sys; 
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Filters import bloom as bl

class TestBloom(unittest.TestCase):
    
    def test_fit(self):
        bf1 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        bf2 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        bf3 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        X = [7800, 4635, 67, 903, 12, 4, 921, 3030, 22, 7010]
        
        y1 = []
        y2 = [True]*len(X)
        y2[0] = False
        y3 = [True] * len(X)
        
        self.assertRaises(ValueError, bf1.fit, X, y1)
        self.assertRaises(ValueError, bf2.fit, X, y2)
        bf3.fit(X, y3)
        self.assertTrue(bf3.fitted)
    
    def test_predict(self):
        bf1 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        bf2 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        X = [7800, 4635, 67, 903, 12, 4, 921, 3030, 22, 7010]
        
        bf1.fit(X)
        
        self.assertTrue((bf1.predict(X) == True).all())
        self.assertRaises(ValueError, bf2.predict, X)
        
    def test_score(self):
        X = [7800, 4635, 67, 903, 12, 4, 921, 3030, 22, 7010]
        bf1 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        bf1.fit(X)
        bf2 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        bf2.fit(X)
        bf3 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        bf4 = bl.BloomFilter(100, [bl.h1, bl.h2, bl.h3])
        bf4.fit(X)
        
        no_X = np.random.uniform(low=0, high=8000, size=1000)
        s = set(no_X)
        for x in X:
            if x in s:
                s.remove(x)
        no_X = np.array(list(s))
        
        y1 = []
        y2 = [False]*len(X)
        y2[0] = True
        
        self.assertRaises(ValueError, bf1.score, X, y1)
        self.assertRaises(ValueError, bf2.score, X, y2)
        self.assertRaises(ValueError, bf3.score, X)
        self.assertTrue(abs(bf4.score(no_X) - bf4.false_positive_prob()) < 1E-2)
    
    
if __name__ == '__main__':
    unittest.main()