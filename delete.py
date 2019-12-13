import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import numpy as np


d1 = {'c1': [1, 2], 'c2': [3, 4]}
df_train = pd.DataFrame(data=d1)
d2 = {'c1': [3, 5], 'c2': [6, 9]}
df_test = pd.DataFrame(data=d2)
dfs = [df_train, df_test]
print(df_train)
print(df_test)

cat_features = ['c1']
encoded_features = []

for df in dfs:
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

df_train = pd.concat([df_train, *encoded_features[:len(cat_features)]], axis=1)
df_test = pd.concat([df_test, *encoded_features[len(cat_features):]], axis=1)

print(df_train)
print(df_test)