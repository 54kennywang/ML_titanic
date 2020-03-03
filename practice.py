import string, random, numpy as np, pandas as pd

a = pd.array([1, 2, 3])
b = pd.array([4, 5, 6])
print(a)
print(b)
print(pd.concat([a, b], sort=True).reset_index(drop=True))