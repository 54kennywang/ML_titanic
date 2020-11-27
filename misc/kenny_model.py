from Advanced_feature_engineering_refactored import advanced_feature_engineer_pd
import pandas as pd
import os
dirname = os.path.dirname(__file__)

# set dataFrame display properties
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load in the train and test datasets
train = pd.read_csv(os.path.join(dirname, './input/train.csv'))
test = pd.read_csv(os.path.join(dirname, './input/test.csv'))

df_train, df_test = advanced_feature_engineer_pd(train, test, one_hot=True)
# unique_values_in_column([df_train, df_test], 5)
# df_train = df_train.drop(['Survived', 'PassengerId'], axis=1)
# df_test = df_test.drop(['PassengerId'], axis=1)
print(df_train.head(1))
print(df_test.head(1))


