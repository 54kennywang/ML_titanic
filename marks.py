import numpy as np
import pandas as pd
import os
dirname = os.path.dirname(__file__)


# solution = os.path.join(dirname, './input/train.csv')
# submission = os.path.join(dirname, './input/test.csv')
# solution = pd.read_csv(solution)
# submission = pd.read_csv(submission)

a = [['10', '1.2', '4.2'], ['15', '70', '0.03'], ['8', '5', '0']]
df1 = pd.DataFrame(a, columns=['one', 'two', 'three'])

b = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
df2 = pd.DataFrame(b, columns=['a', 'b', 'c'])

df = pd.concat([a[['one']], b[['a']]], axis=1)
print(df)





