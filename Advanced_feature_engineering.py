# bin-ing is based on visualization (bin by proper gap + gropu by characteristics)
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
deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'M': 0}
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

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

# print(df_all[df_all['Cabin'].isna()])
# print(df_all[df_all['Cabin'].notnull()])

df_all['Deck'] = df_all['Cabin'].apply(lambda carbin: carbin[0] if pd.notnull(carbin) else 'M')
# outFile = os.path.join(dirname, './decks.csv')
# df_all.to_csv(outFile, index=False)
# use Tableau on decks.csv for deck visualization

# There is one person on the boat deck in the T cabin and he is a 1st class passenger. T cabin passenger has the closest resemblance to A deck passengers, so he is grouped in A deck.
print(df_all.loc[(df_all['Deck'] == 'T')])
idx = df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx, 'Deck'] = 'A'
print(df_all.loc[[339]]) # confirm this row is changed to Deck = 'A'

"""
By Tableau visualization:
1. A, B and C decks are labeled as ABC because all of them have only 1st class passengers.
2. D and E decks are labeled as DE because both of them have similar passenger class distribution and same survival rate.
3. F and G decks are labeled as FG because of the previous reasons.
4. M deck doesn't need to be grouped with other decks because it is very different from others and has the lowest survival rate.

df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')

df_all['Deck'].value_counts()
"""

# Dropping the Cabin feature
df_all.drop(['Cabin'], inplace=True, axis=1)

""" no more missingness
"""
df_train, df_test = divide_df(df_all)
df_train.name = 'Training Set'
df_test.name = 'Test Set'
dfs = [df_train, df_test]
for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)

dfs = categorical_to_ordinal(dfs, deck_mapping, 'Deck')
df_train, df_test = dfs[0], dfs[1]
df_all = concat_df(df_train, df_test)
dfs = [df_train, df_test]

def Pearson_Correlation_of_Features(train):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()

def add_title_col(dfs):
    # this will loop through all rows in both train and test, it updates train_df, test_df as well
    print('=== extract title from name ===')
    for dataset in dfs:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # use regex to generate Title
    return dfs

dfs = add_title_col(dfs)
df_train, df_test = dfs[0], dfs[1]

# ['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer']
def caregorize_title(dfs):
    for dataset in dfs:
        dataset['Title'] = dataset['Title'].replace(['Col', 'Lady', 'Countess','Capt', 'Col' 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Don'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    return dfs

dfs = caregorize_title(dfs)
df_train, df_test = dfs[0], dfs[1]

def add_name_length(dfs):
    # this will loop through all rows in both train and test, it updates train_df, test_df as well
    print('=== extract title length from name ===')
    for dataset in dfs:
        dataset['nameLen'] = dataset['Name'].apply(len)
    return dfs

dfs = add_name_length(dfs)
df_train, df_test = dfs[0], dfs[1]

dfs = categorical_to_ordinal(dfs, title_mapping, 'Title')
df_train, df_test = dfs[0], dfs[1]

def create_FamilySize(dfs):
    print('=== Create family size ===')
    for dataset in dfs:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    return dfs

dfs = create_FamilySize(dfs)
df_train, df_test = dfs[0], dfs[1]


def create_ticketFreq(dfs):
    print('=== Create ticketFreq ===')
    for dataset in dfs:
        dataset['ticketFreq'] = dataset.groupby('Ticket')['Ticket'].transform('count')
    return dfs

dfs = create_ticketFreq(dfs)
df_train, df_test = dfs[0], dfs[1]

#
def create_IsAlone(dfs):
    print('=== Create IsAlone ===')
    for dataset in dfs:
        dataset['IsAlone'] = 0
        dataset.loc[(dataset['FamilySize'] == 1) & (dataset['ticketFreq'] == 1), 'IsAlone'] = 1
    return dfs

dfs = create_IsAlone(dfs)
df_train, df_test = dfs[0], dfs[1]


"""
[0, 5]
[5, 25]
[25, 30]
[30, 35]
[35, 40]
[40, 65]
[65, 80]
"""
def convert_age_to_ordinal_based_on_bins(dfs):
    print('=== convert age to ordinal value based on age bins ===')
    for dataset in dfs:
        dataset.loc[ dataset['Age'] <= 5, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 25), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 35), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 40), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 65), 'Age'] = 5
        dataset.loc[ dataset['Age'] > 65, 'Age'] = 6
    return dfs

dfs = convert_age_to_ordinal_based_on_bins(dfs)
df_train, df_test = dfs[0], dfs[1]

"""
[0, 15]
[15, 75]
[75, 120]
[120, 135]
[135, 510]
"""
def convert_fare_to_ordinal_based_on_bins(dfs):
    print('=== convert fare to ordinal value based on fare bins ===')
    for dataset in dfs:
        dataset.loc[ dataset['Fare'] <= 15, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 75), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 75) & (dataset['Fare'] <= 120), 'Fare'] = 2
        dataset.loc[(dataset['Fare'] > 120) & (dataset['Fare'] <= 135), 'Fare'] = 3
        dataset.loc[ dataset['Fare'] > 135, 'Fare'] = 4
        dataset['Fare'] = dataset['Fare'].astype(int)
    return dfs

dfs = convert_fare_to_ordinal_based_on_bins(dfs)
df_train, df_test = dfs[0], dfs[1]


def drop_col(df, col_arr, axis=1):
    print('=== Drop useless cols ===')
    df = df.drop(col_arr, axis=axis)
    return df

df_train = drop_col(df_train, 'Name')
df_train = drop_col(df_train, 'Ticket')
df_test = drop_col(df_test, 'Name')
df_test = drop_col(df_test, 'Ticket')
df_all = concat_df(df_train, df_test)
dfs = [df_train, df_test]
print(df_train.head(3))
print(df_test.tail(3))


# one-hot vec


def output(df):
    outFile = os.path.join(dirname, './output/data_to_be_analyzed.csv')
    df.to_csv(outFile, index=False)

# output(df_train)
# imputation
# drop nameLen
