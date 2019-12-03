import os
import pandas as pd
import numpy as np
import random as rnd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# set dataFrame display properties
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# data input
train_input = '/Users/kennywang/Documents/study/self/ML_titanic/input/train.csv'
test_input = '/Users/kennywang/Documents/study/self/ML_titanic/input/test.csv'

# mappings
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
sex_mapping = {'female': 1, 'male': 0}
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}

# get raw data from input files
def set_raw_data(train, test):
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    combine = [train_df, test_df] # reference copy
    return train_df, test_df, combine

# see column names of dataframe
def get_col_names(df):
    print('=== Col names ===')
    print(df.columns.values)

# preview the data
def view_head_data(df, num = 3):
    print('=== First ' + str(num) + ' rows ===')
    print(df.head(num))

# preview the data
def view_tail_data(df, num = 3):
    print('=== Last ' + num + ' rows ===')
    print(df.tail(num))

def view_data_info(df):
    print('=== info ===')
    df.info()

# null percentage of cols in a df
def get_null_percentage(df):
    print('=== Null percentage ===')
    print(df.isna().sum() / df.shape[0])

# basic df col stats
def get_col_stats(df):
    print('=== Col stats ===')
    print(df.describe())
    print(df.describe(include=['O']))

"""
# diff domain associated with survival rate
print('=== Different domain comparision ===')
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*40)
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*40)
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*40)
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*40)
print(train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*40)
# print(train_df[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*40)
# print(train_df[["Age", "Survived"]].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False))
"""
# cross info
# print(pd.crosstab(train_df['Title'], train_df['Sex']))


# drop useless cols (array) of df
def drop_col(df, col_arr, axis=1):
    print('=== Drop useless cols ===')
    df = df.drop(col_arr, axis=axis)
    return df

# project-specific
def add_title_col(combine):
    # this will loop through all rows in both train and test, it updates train_df, test_df as well
    print('=== extract title from name ===')
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # use regex to generate Title
    return combine

# categorize title to Miss/Mrs/Rare/Master/Mr
def caregorize_title(combine):
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Col', 'Lady', 'Countess','Capt', 'Col' 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Don'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    return combine

# convert the categorical titles to ordinal
def categorical_to_ordinal(combine, mapping, col):
    for dataset in combine:
        for key, val in mapping.items():
            dataset[col] = dataset[col].replace(key, int(val))
    return combine


# project-specific
def fill_missing_age(combine):
    '''
    More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender,
    and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations.
    So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
    '''
    print('=== Fill in missing age ===')
    guess_ages = np.zeros((2,3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                # all age data of rows where Sex=i and Pclass=(j+1), for example:
                # 1   47.00
                # 4   22.00
                # 6   30.00
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
                # print('(',i,j,')',guess_df)

                age_guess = guess_df.median()
                # age_guess = guess_df.mean()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
                # df.loc: access a group of rows and columns by label(s) or a boolean array.
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)
    return combine

# a different way to generate missing age
def fill_missing_age2(combine):
    for dataset in combine:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()

        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    return combine


def group_age_into_bins(df):
    # print(train_df.describe()) shows 0.0 <= Age <= 80.0
    print('=== Group age into bins ===')
    # Use cut when you need to segment and sort data values into bins, here to put age into 5 bins
    df['AgeBand'] = pd.cut(df['Age'], 5)
    return df

def convert_age_to_ordinal_based_on_age_bins(combine):
    print('=== convert age to ordinal value based on age bins ===')
    for dataset in combine:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    return combine

def fill_missing_fare(df):
    print('=== Fill in missing fare in test ===')
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    return df

def group_fare_into_bins(df):
    print('=== Group fare into bins ===')
    # print(train_df.describe()) shows 0.0 <= Fare <= 512.329200
    # Use cut when you need to segment and sort data values into bins, here to put fare into 4 bins
    df['FareBand'] = pd.cut(df['Fare'], 4)
    return df

def convert_fare_to_ordinal_based_on_fare_bins(combine):
    print('=== convert fare to ordinal value based on fare bins ===')
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    return combine

def fill_missing_embarked(df, combine):
    print('=== Fill in missing embarked in train ===')
    # only 2 missing Embarked in train_df, simply replace them with the most frequent one
    freq_port = df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    return combine

def create_FamilySize(combine):
    print('=== Create family size ===')
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    return combine

def create_IsAlone(combine):
    print('=== Create IsAlone ===')
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    return combine

def update_df(combine):
    train_df = combine[0]
    test_df = combine[1]
    return train_df, test_df

# Model
"""
print('=== Model ===')
X_train = train_df.drop(["PassengerId", "Survived"], axis=1) # (891, 10)
Y_train = train_df["Survived"] # (891,)
X_test  = test_df.drop("PassengerId", axis=1).copy() # (418, 10)

# i, c = np.where(X_train == 'Col')
# print ((X_train.index[i][0], X_train.columns[c][0]))
X_train = X_train.astype(int)
Y_train = Y_train.astype(int)
X_test = X_test.astype(int)


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("LogisticRegression:", acc_log)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("DecisionTree", acc_decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("RandomForest", acc_random_forest)

# submission
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": Y_pred})
# submission.to_csv('../output/submission.csv', index=False)
# print(submission)
"""

def Model(train_input, test_input):
    train_df, test_df, combine = set_raw_data(train_input, test_input)

    # get_col_names(train_df)
    # get_col_names(test_df)
    #
    # view_head_data(train_df, 3)
    # view_head_data(test_df, 3)
    #
    # get_null_percentage(train_df)
    # get_null_percentage(test_df)
    #
    # get_col_stats(train_df)
    # get_col_stats(test_df)

    # extract and add Title from Name
    combine = add_title_col(combine)

    # convert Title to ordinal
    combine = caregorize_title(combine)
    combine = categorical_to_ordinal(combine, title_mapping, 'Title')
    train_df, test_df = update_df(combine)
    # remove Name col
    train_df = drop_col(train_df, ['Name'])
    test_df = drop_col(test_df, 'Name')
    combine = [train_df, test_df]

    # convert Sex to ordinal
    combine = categorical_to_ordinal(combine, sex_mapping, 'Sex')
    train_df, test_df = update_df(combine)

    # Fill in missing age
    combine = fill_missing_age2(combine)
    train_df, test_df = update_df(combine)

    # Group age into bins
    train_df = group_age_into_bins(train_df)
    test_df = group_age_into_bins(test_df)
    combine = [train_df, test_df]

    # convert age to ordinal based on age bins
    combine = convert_age_to_ordinal_based_on_age_bins(combine)
    train_df, test_df = update_df(combine)

    # Group fare into bins
    train_df = group_fare_into_bins(train_df)
    test_df = group_fare_into_bins(test_df)
    combine = [train_df, test_df]

    # Fill in missing fare
    test_df = fill_missing_fare(test_df)
    combine = [train_df, test_df]

    # convert fare to ordinal based on fare bins
    combine = convert_fare_to_ordinal_based_on_fare_bins(combine)
    train_df, test_df = update_df(combine)

    # Fill in missing Embarked
    combine = fill_missing_embarked(train_df, combine)
    train_df, test_df = update_df(combine)

    # Convert Embarked to ordinal
    combine = categorical_to_ordinal(combine, embarked_mapping, 'Embarked')
    train_df, test_df = update_df(combine)

    # create FamilySize
    combine = create_FamilySize(combine)
    train_df, test_df = update_df(combine)

    # create IsAlone
    combine = create_IsAlone(combine)
    train_df, test_df = update_df(combine)

    # drop useless columns
    train_df = drop_col(train_df, ['Cabin', 'Ticket', 'AgeBand', 'FareBand'])
    test_df = drop_col(test_df, ['Cabin', 'Ticket', 'AgeBand', 'FareBand'])
    combine = [train_df, test_df]

    # view_head_data(train_df, 3)
    # view_head_data(test_df, 3)
    #
    # get_null_percentage(train_df)
    # get_null_percentage(test_df)

    print('=== Model ===')
    X_train = train_df.drop(["PassengerId", "Survived"], axis=1)  # (891, 10)
    Y_train = train_df["Survived"]  # (891,)
    X_test = test_df.drop("PassengerId", axis=1).copy()  # (418, 10)

    X_train = X_train.astype(int)
    Y_train = Y_train.astype(int)
    X_test = X_test.astype(int)

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print("LogisticRegression:", acc_log)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print("DecisionTree", acc_decision_tree)


if __name__== "__main__":
    Model(train_input, test_input)