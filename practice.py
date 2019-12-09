import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import numpy as np


d1 = {'col': [1, 2], 'cc': [3, 4]}
df_train = pd.DataFrame(data=d1)
d2 = {'col': [3, 5], 'cc': [6, 9]}
df_test = pd.DataFrame(data=d2)
print(df_train)
print(df_test)
dfs = [df_train, df_test]

cat_features = ['col', 'cc']
encoded_features = []  # one-hot vec of train+test
maxs = []
for i in range(len(cat_features)):
    maxs.append(-1)
mins = []
for i in range(len(cat_features)):
    mins.append(100)
for df in dfs:
    for feature in cat_features:
        ma = df[feature].max()
        if(maxs[cat_features.index(feature)] < ma):
            maxs[cat_features.index(feature)] = ma
        mi = df[feature].min()
        if (mins[cat_features.index(feature)] > mi):
            mins[cat_features.index(feature)] = mi
print(maxs)
print(mins)
print('====')
print(df_train['col'].values.tolist())

x = df_train['col'].values.tolist()
y = []
for i in x:
    t = [0] * 5
    t[i] = 1
    y.append(t)
print(y)

print('====')

for df in dfs:
    for feature in cat_features:
        # print(np.append(df[feature].values, np.array([0, 0, 0])))
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()  # data transformation to one-hot
        # encoded_feat = OneHotEncoder().fit_transform(np.append(df[feature].values, np.array([0, 0, 0])).reshape(-1, 1)).toarray()  # data transformation to one-hot
        n = df[feature].nunique()  # find num of unique vals of that feature
        # print('n = ', n)
        # n = 5
        cols = ['{}_{}'.format(feature, n) for n in range(0, n)]  # generate new col names based on n
        print('encoded_feat: ', encoded_feat)
        print('cols: ', cols)
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)  # one-hot data pair with new cols from above
        encoded_df.index = df.index
        encoded_features.append(encoded_df)
df_train = pd.concat([df_train, *encoded_features[:len(cat_features)]], axis=1)
df_test = pd.concat([df_test, *encoded_features[len(cat_features):]], axis=1)


print(df_train)
print(df_test)