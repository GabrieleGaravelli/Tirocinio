import math


S = {7800, 4635, 67, 903, 12, 4, 921, 3030, 22, 7010}
t = {7800, 9, 564, 4635, 345, 67, 903, 7, 9000, 12, 500, 4, 921, 12, 3030, 2323, 1763, 22, 7010, 232, 6000, 5760, 111, 93, 6466, 389, 743, 8370, 5417}
n = []
result = set()

ts = 41

for x in range(ts):
    n.append(0)

def h1(x):
    return x % ts

def h2(x):
    return math.floor(ts * (x * 0.77 % 1))

def h3(x):
    return hash(str(x)) % ts

for x in S:
    n[h1(x)] = 1
    n[h2(x)] = 1
    n[h3(x)] = 1

for x in t:
    if n[h1(x)] == 1 and n[h2(x)] == 1 and n[h3(x)] == 1:
        result.add(x)


print(result.issuperset(S))
print(result)






