import string, random, numpy as np, pandas as pd

s = pd.Series(list('abcae'))

print(pd.get_dummies(s))