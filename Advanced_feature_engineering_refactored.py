# bin-ing is based on visualization (bin by proper gap + gropu by characteristics)
import numpy as np
import pandas as pd
# set dataFrame display properties
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
dirname = os.path.dirname(__file__)

import warnings
warnings.filterwarnings('ignore')

sex_mapping = {'female': 1, 'male': 0}
embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}
deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'M': 0}
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

df_train = pd.read_csv(os.path.join(dirname, './input/train.csv'))
df_test = pd.read_csv(os.path.join(dirname, './input/test.csv'))
dfs = [df_train, df_test]

def display_missingness(dfs):
    print('===display missingness===')
    for df in dfs:
        for col in df.columns.tolist():
            print( '{} column missing values: {}/{} = {}'.format(col, df[col].isnull().sum(), df.shape[0], df[col].isnull().sum()/df.shape[0]))
        print('-------')
    print('===================\n')

def concat_df(df_train, df_test):
    # Returns a concatenated df of training and test set on axis 0
    return pd.concat([df_train, df_test], sort=True).reset_index(drop=True)

# convert the categorical titles to ordinal
def categorical_to_ordinal(dfs, mapping, col):
    print('===categorical to ordinal for '+col+'===')
    for dataset in dfs:
        for key, val in mapping.items():
            dataset[col] = dataset[col].replace(key, int(val))
        dataset[col] = dataset[col].astype('Int64')
    return dfs

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

# find absolute value of correlation between given col and the rest of columns, missing values are ignored, only for numerical
# suitable for finding correlation in order to decide how to fill in missing-ness
def correlation(df, col, abs):
    print('===correlation with '+col+'===')
    # df.coor(): Compute pairwise correlation of columns, excluding NA/null values.
    if (abs == True):
        df_all_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    else:
        df_all_corr = df.corr().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    print(df_all_corr[df_all_corr['Feature 1'] == col])
    print('===================\n')

def fill_missing_age(df_all):
    print('===fill missing age===')
    df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    return df_all

# Both of those passengers are female, upper class and same ticket number. This means, they know each other and embarked from the same port together.
# When I googled Stone, Mrs. George Nelson (Martha Evelyn), I found that she embarked from S (Southampton) with her maid Amelie Icard, in this page Martha Evelyn Stone: Titanic Survivor. Mrs Stone boarded the Titanic in Southampton on 10 April 1912 and was travelling in first class with her maid Amelie Icard. She occupied cabin B-28. This is the information needed and case closed for Embarked feature.
# Filling the missing values in Embarked with S
def fill_missing_embarked(df_all):
    print('===fill missing embarked===')
    df_all['Embarked'] = df_all['Embarked'].fillna(embarked_mapping['S'])
    return df_all

def fill_missing_fare(df_all):
    print('===fill missing fare===')
    median_fare = df_all.loc[
        (df_all['Pclass'] == 3) & (df_all['Parch'] == 0) & (df_all['Sex'] == 0) & (df_all['SibSp'] == 0)].Fare.median()
    df_all['Fare'] = df_all['Fare'].fillna(median_fare)
    return df_all

def create_deck(df_all):
    print('===create deck col===')
    df_all['Deck'] = df_all['Cabin'].apply(lambda carbin: carbin[0] if pd.notnull(carbin) else 'M')
    return df_all

def drop_col(df, col_arr, axis=1):
    print('=== Drop useless cols ===')
    df = df.drop(col_arr, axis=axis)
    return df

def add_title_col(dfs):
    # this will loop through all rows in both train and test, it updates train_df, test_df as well
    print('=== extract title from name ===')
    for dataset in dfs:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # use regex to generate Title
    return dfs

def caregorize_title(dfs):
    print('=== caregorize titles to Rare/Miss/Mrs ===')
    for dataset in dfs:
        dataset['Title'] = dataset['Title'].replace(['Col', 'Lady', 'Countess','Capt', 'Col' 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Don'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    return dfs

def create_FamilySize(dfs):
    print('=== Create family size ===')
    for dataset in dfs:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    return dfs

def create_ticketFreq(dfs):
    print('=== Create ticketFreq ===')
    for dataset in dfs:
        dataset['ticketFreq'] = dataset.groupby('Ticket')['Ticket'].transform('count')
    return dfs

def create_IsAlone(dfs):
    print('=== Create IsAlone ===')
    for dataset in dfs:
        dataset['IsAlone'] = 0
        dataset.loc[(dataset['FamilySize'] == 1) & (dataset['ticketFreq'] == 1), 'IsAlone'] = 1
    return dfs


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
        dataset['Age'] = dataset['Age'].astype(int)
    return dfs

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

def add_name_length(dfs):
    # this will loop through all rows in both train and test, it updates train_df, test_df as well
    print('=== extract title length from name ===')
    for dataset in dfs:
        dataset['nameLen'] = dataset['Name'].apply(len)
    return dfs

def advanced_feature_engineer(df_train, df_test):
    dfs = [df_train, df_test]
    df_all = concat_df(df_train, df_test)

    # print(df_train.head(3))
    display_missingness(dfs)

    dfs = categorical_to_ordinal(dfs, sex_mapping, 'Sex')
    dfs = categorical_to_ordinal(dfs, embarked_mapping, 'Embarked')
    df_all = concat_df(df_train, df_test)

    correlation(df_all, 'Age', True)  # => Median age of Pclass groups is the best choice because of its high correlation with Age
    df_all = fill_missing_age(df_all)
    df_train, df_test = divide_df(df_all)
    dfs = [df_train, df_test]
    display_missingness(dfs)

    print(df_all[df_all['Embarked'].isnull()])
    correlation(df_all, 'Embarked', True)  # => this does not give much info about which col is closely related to 'Embarked'
    df_all = fill_missing_embarked(df_all)
    df_train, df_test = divide_df(df_all)
    dfs = [df_train, df_test]
    display_missingness(dfs)

    print(df_all[df_all['Fare'].isnull()])
    correlation(df_all, 'Fare', True)  # => highly associated with 'Pclass', 'Parch', 'SibSp', 'Sex'
    df_all = fill_missing_fare(df_all)
    df_train, df_test = divide_df(df_all)
    dfs = [df_train, df_test]
    display_missingness(dfs)

    df_all = create_deck(df_all)
    # There is one person on the boat deck in the T cabin and he is a 1st class passenger. T cabin passenger has the closest resemblance to A deck passengers, so he is grouped in A deck.
    idx = df_all[df_all['Deck'] == 'T'].index
    df_all.loc[idx, 'Deck'] = 'A'
    # print(df_all.loc[[339]])  # confirm this row is changed to Deck = 'A'
    df_train, df_test = divide_df(df_all)
    dfs = [df_train, df_test]
    # print(df_train.head(3))

    dfs = categorical_to_ordinal(dfs, deck_mapping, 'Deck')
    df_all = concat_df(df_train, df_test)
    # print(df_train.head(3))

    dfs = add_title_col(dfs)
    df_all = concat_df(df_train, df_test)
    # print(df_train.head(3))

    dfs = caregorize_title(dfs)
    df_all = concat_df(df_train, df_test)
    # print(df_train.head(3))

    dfs = categorical_to_ordinal(dfs, title_mapping, 'Title')
    df_all = concat_df(df_train, df_test)
    # print(df_train.head(3))

    dfs = create_FamilySize(dfs)
    # print(df_train.head(3))

    dfs = create_ticketFreq(dfs)
    # print(df_train.head(3))

    dfs = create_IsAlone(dfs)
    # print(df_train.head(3))

    dfs = convert_age_to_ordinal_based_on_bins(dfs)
    # print(df_train.head(3))

    dfs = convert_fare_to_ordinal_based_on_bins(dfs)
    # print(df_train.head(3))

    dfs = add_name_length(dfs)
    print(df_train.head(3))

    df_train = drop_col(df_train, ['Cabin', 'Name', 'Ticket'])
    df_test = drop_col(df_test, ['Cabin', 'Name', 'Ticket'])
    df_all = concat_df(df_train, df_test)
    dfs = [df_train, df_test]
    print(df_train.head(3))
    print(df_test.head(3))
    display_missingness(dfs)
    print('=== no more missingness ==')

    # Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
    y_train = df_train['Survived'].ravel()
    train = df_train.drop(['Survived', 'PassengerId'], axis=1)
    test = df_test.drop(['PassengerId'], axis=1)
    print(train.head(3))
    print(test.head(3))
    tr = train
    x_train = train.values  # Creates an array of the train data
    x_test = test.values  # Creats an array of the test data
    return x_train, y_train, x_test, tr

if __name__== "__main__":
    advanced_feature_engineer(df_train, df_test)