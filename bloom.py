import math


class Bloom():

    k=3

    def __init__(self, ts, S):
        self.ts = ts
        self.S = S
        self.n = []

        for x in range(ts):
            self.n.append(0)

        for x in S:
            self.n[h1(x, ts)] = 1
            self.n[h2(x, ts)] = 1
            self.n[h3(x, ts)] = 1
        
    def filter(self, T):
        self.result = set()
        for x in T:
            if self.n[h1(x, self.ts)] == 1 and self.n[h2(x, self.ts)] == 1 and self.n[h3(x, self.ts)] == 1:
                self.result.add(x)
        fp = (1 - math.e**((-self.k)*len(self.S)/self.ts))**self.k
        print(self.result.issuperset(self.S))
        print('Theoretical false positive rate: {}'.format(fp))
        print('Actual false positive rate: {}'.format((len(self.result) - len(self.S))/(len(T) - len(self.S))))
        return self.result 


def h1(x, ts):
    return x % ts

def h2(x, ts):
    return math.floor(ts * (x * 0.77 % 1))

def h3(x, ts):
    return hash(str(x)) % ts







