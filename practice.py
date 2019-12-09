import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


d1 = {'col': [1, 2]}
df_train = pd.DataFrame(data=d1)
d2 = {'col': [3, 5]}
df_test = pd.DataFrame(data=d2)
print(df_train)
print(df_test)
dfs = [df_train, df_test]

cat_features = ['col']
encoded_features = []  # one-hot vec of train+test
maxs = []
for i in range(len(cat_features)):
    maxs.append(-1)
mins = []
for i in range(len(cat_features)):
    mins.append(100)
print(maxs)
print(mins)
print('&&&&')
for df in dfs:
    for feature in cat_features:
        ma = df[feature].max()
        if(maxs[cat_features.index(feature)] < ma):
            maxs[cat_features.index(feature)] = ma
        mi = df[feature].min()
        if (mins[cat_features.index(feature)] > mi):
            mins[cat_features.index(feature)] = mi
print(df_train['col'].values)



for df in dfs:
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()  # data transformation to one-hot
        n = df[feature].nunique()  # find num of unique vals of that feature
        cols = ['{}_{}'.format(feature, n) for n in range(0, n)]  # generate new col names based on n
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)  # one-hot data pair with new cols from above
        encoded_df.index = df.index
        encoded_features.append(encoded_df)
df_train = pd.concat([df_train, *encoded_features[:len(cat_features)]], axis=1)
df_test = pd.concat([df_test, *encoded_features[len(cat_features):]], axis=1)


print(df_train)
print(df_test)