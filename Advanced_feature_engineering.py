import numpy as np
import pandas as pd
# set dataFrame display properties
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
dirname = os.path.dirname(__file__)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import string
import warnings
warnings.filterwarnings('ignore')
SEED = 42

sex_mapping = {'female': 1, 'male': 0}
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set on axis 0
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

# convert the categorical titles to ordinal
def categorical_to_ordinal(dfs, mapping, col):
    for dataset in dfs:
        for key, val in mapping.items():
            dataset[col] = dataset[col].replace(key, int(val))
        dataset[col] = dataset[col].astype('Int64')
    return dfs


df_train = pd.read_csv(os.path.join(dirname, './input/train.csv'))
df_test = pd.read_csv(os.path.join(dirname, './input/test.csv'))
df_all = concat_df(df_train, df_test)
# print(df_train.head(3))
# print(df_test.head(3))

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set'

dfs = [df_train, df_test]

num_train = df_train.shape[0]
num_test = df_test.shape[0]

# display missing-ness
def display_missing(df):
    for col in df.columns.tolist():
        print( '{} column missing values: {}/{} = {}'.format(col, df[col].isnull().sum(), df.shape[0], df[col].isnull().sum()/df.shape[0]))
    print('\n')

print('===Raw data missingness===')
for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)

dfs = categorical_to_ordinal(dfs, sex_mapping, 'Sex')
dfs = categorical_to_ordinal(dfs, embarked_mapping, 'Embarked')
df_train, df_test = dfs[0], dfs[1]
df_all = concat_df(df_train, df_test)
dfs = [df_train, df_test]
print(df_train.head(3))
print(df_test.head(3))

print('===Missingness after categorical to ordinal===') # => categorical_to_ordinal doesn't drop NULL, simple ignore
for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)

# find absolute value of correlation between given col and the rest of columns, missing values are ignored, only for numerical
# suitable for finding correlation in order to decide how to fill in missing-ness
def correlation(df, col, abs):
    # df.coor(): Compute pairwise correlation of columns, excluding NA/null values.
    if (abs == True):
        df_all_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    else:
        df_all_corr = df.corr().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    print(df_all_corr[df_all_corr['Feature 1'] == col])

correlation(df_all, 'Age', True) # => Median age of Pclass groups is the best choice because of its high correlation with Age
print()


age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']
for pclass in range(1, 4):
    for sex in ['female', 'male']:
        sex_numerical = sex_mapping[sex]
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex_numerical][pclass]))
print('Median age of all passengers: {}'.format(df_all['Age'].median()))
# Filling the missing values in Age with the medians of Sex and Pclass groups
# x.apply(func) # x is the input of func
# lambda arg: formula # arg is df here
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


print(df_all[df_all['Embarked'].isnull()])
correlation(df_all, 'Embarked', True) # => this does not give much info about which col is closely related to 'Embarked'
# Both of those passengers are female, upper class and same ticket number. This means, they know each other and embarked from the same port together.
# When I googled Stone, Mrs. George Nelson (Martha Evelyn), I found that she embarked from S (Southampton) with her maid Amelie Icard, in this page Martha Evelyn Stone: Titanic Survivor. Mrs Stone boarded the Titanic in Southampton on 10 April 1912 and was travelling in first class with her maid Amelie Icard. She occupied cabin B-28. This is the information needed and case closed for Embarked feature.
# Filling the missing values in Embarked with S
df_all['Embarked'] = df_all['Embarked'].fillna(embarked_mapping['S'])
# print(df_all.loc[[61]]) # to confirm null is filled

print(df_all[df_all['Fare'].isnull()])
correlation(df_all, 'Fare', False) # => highly associated with 'Pclass', 'Parch', 'SibSp'
# print(df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]) # from kernal
median_fare = df_all.loc[(df_all['Pclass'] == 3) & (df_all['Parch'] == 0) & (df_all['Sex'] == 0) & (df_all['SibSp'] == 0)].Fare.median() # # median_fare = df_all.query('Pclass == 3 & Parch == 0 & Sex == 0 & SibSp == 0').Fare.median()
print(median_fare) # my own way
df_all['Fare'] = df_all['Fare'].fillna(median_fare)
# print(df_all.loc[[1043]]) # to confirm null is filled

print('-----start from here-------')
# print(df_all[df_all['Cabin'].isna()])
# print(df_all[df_all['Cabin'].notnull()])

df_all['Deck'] = df_all['Cabin'].apply(lambda carbin: carbin[0] if pd.notnull(carbin) else 'M')
outFile = os.path.join(dirname, './output/decks.csv')
df_all.to_csv(outFile, index=False)

