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

import seaborn as sns
import matplotlib.pyplot as plt
import os
dirname = os.path.dirname(__file__)

# set dataFrame display properties
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# data input
train_input = os.path.join(dirname, './input/train.csv')
test_input = os.path.join(dirname, './input/test.csv')

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
def get_null_percentage(combine):
    print('=== Null percentage ===')
    for dataset in combine:
        print(dataset.isna().sum() / dataset.shape[0])
        print('-------------------')

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

# project-specific
def add_name_length(combine):
    # this will loop through all rows in both train and test, it updates train_df, test_df as well
    print('=== extract title length from name ===')
    for dataset in combine:
        dataset['nameLen'] = dataset['Name'].apply(len)
    return combine

# categorize title to Miss/Mrs/Rare/Master/Mr
def categorize_title(combine):
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


def group_age_into_bins(combine):
    # print(train_df.describe()) shows 0.0 <= Age <= 80.0
    print('=== Group age into bins ===')
    for dataset in combine:
        dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
    # Use cut when you need to segment and sort data values into bins, here to put age into 5 bins
    return combine

def convert_age_to_ordinal_based_on_age_bins(combine):
    print('=== convert age to ordinal value based on age bins ===')
    for dataset in combine:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    return combine

def fill_missing_fare(combine):
    print('=== Fill in missing fare in test ===')
    combine[1]['Fare'].fillna(combine[1]['Fare'].dropna().median(), inplace=True)
    return combine

def group_fare_into_bins(combine):
    print('=== Group fare into bins ===')
    for dataset in combine:
        dataset['FareBand'] = pd.cut(dataset['Fare'], 4)
    # print(train_df.describe()) shows 0.0 <= Fare <= 512.329200
    # Use cut when you need to segment and sort data values into bins, here to put fare into 4 bins
    return combine

def convert_fare_to_ordinal_based_on_fare_bins(combine):
    print('=== convert fare to ordinal value based on fare bins ===')
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
    return combine

def fill_missing_embarked(combine):
    print('=== Fill in missing embarked in train ===')
    # only 2 missing Embarked in train_df, simply replace them with the most frequent one
    freq_port = combine[0].Embarked.dropna().mode()[0]
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

def Pearson_Correlation_of_Features(train_df):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train_df.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()

def feature_importance(cls, X_train, Y_train):
    importance = (cls.fit(X_train, Y_train).feature_importances_).tolist()
    titles = X_train.columns.values.tolist()
    return titles, importance

def submission(test_df, Y_pred):
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    outFile = os.path.join(dirname, './output/FeatureEngineeringSubmission.csv')
    submission.to_csv(outFile, index=False)

def Model(train_input, test_input):
    train_df, test_df, combine = set_raw_data(train_input, test_input)

    # get_col_names(train_df)
    # get_col_names(test_df)
    #
    # view_head_data(train_df, 3)
    # view_head_data(test_df, 3)
    #
    # get_null_percentage(combine)
    #
    get_col_stats(train_df)
    # get_col_stats(test_df)

    # extract and add Title from Name
    add_title_col(combine)

    # convert Title to ordinal
    categorize_title(combine)
    categorical_to_ordinal(combine, title_mapping, 'Title')

    # add nameLen col
    add_name_length(combine)

    # remove Name col
    train_df = drop_col(train_df, ['Name'])
    test_df = drop_col(test_df, 'Name')
    combine = [train_df, test_df]

    # convert Sex to ordinal
    categorical_to_ordinal(combine, sex_mapping, 'Sex')

    # Fill in missing age
    fill_missing_age2(combine)

    # Group age into bins
    group_age_into_bins(combine)

    # convert age to ordinal based on age bins
    convert_age_to_ordinal_based_on_age_bins(combine)

    # Group fare into bins
    group_fare_into_bins(combine)

    # Fill in missing fare
    fill_missing_fare(combine)

    # convert fare to ordinal based on fare bins
    convert_fare_to_ordinal_based_on_fare_bins(combine)

    # Fill in missing Embarked
    fill_missing_embarked(combine)

    # Convert Embarked to ordinal
    categorical_to_ordinal(combine, embarked_mapping, 'Embarked')

    # create FamilySize
    create_FamilySize(combine)

    # create IsAlone
    create_IsAlone(combine)

    # drop useless columns
    train_df = drop_col(train_df, ['Cabin', 'Ticket', 'AgeBand', 'FareBand'])
    test_df = drop_col(test_df, ['Cabin', 'Ticket', 'AgeBand', 'FareBand'])
    combine = [train_df, test_df]

    # view_head_data(train_df, 3)
    # view_head_data(test_df, 3)
    #
    # get_null_percentage(combine)
    # Pearson_Correlation_of_Features(train_df)

    print('=== Model ===')
    X_train = train_df.drop(["PassengerId", "Survived"], axis=1)  # (891, 10)
    Y_train = train_df["Survived"]  # (891,)
    X_test = test_df.drop("PassengerId", axis=1).copy()  # (418, 10)

    X_train = X_train.astype(int)
    Y_train = Y_train.astype(int)
    X_test = X_test.astype(int)

    decision_tree = DecisionTreeClassifier()
    # feature importance
    titles, importance = feature_importance(decision_tree, X_train, Y_train)
    print(titles)
    print(importance)

    decision_tree.fit(X_train, Y_train) # 96.3
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print("DecisionTree", acc_decision_tree)
    # submission(test_df, Y_pred)

    """
    logreg = LogisticRegression() # 80.92
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print("LogisticRegression:", acc_log)

    random_forest = RandomForestClassifier() # 95.4
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print("RandomForest", acc_random_forest)

    KNeighbors= KNeighborsClassifier() # 85.86
    KNeighbors.fit(X_train, Y_train)
    Y_pred = KNeighbors.predict(X_test)
    acc_KNeighbors = round(KNeighbors.score(X_train, Y_train) * 100, 2)
    print("KNeighbors", acc_KNeighbors)

    Gaussian = GaussianNB() # 80.25
    Gaussian.fit(X_train, Y_train)
    Y_pred = Gaussian.predict(X_test)
    acc_Gaussian = round(Gaussian.score(X_train, Y_train) * 100, 2)
    print("Gaussian", acc_Gaussian)

    # Percept = Perceptron()
    # Percept.fit(X_train, Y_train)
    # Y_pred = Percept.predict(X_test)
    # acc_rPercept = round(Percept.score(X_train, Y_train) * 100, 2)
    # print("Percept", Percept)

    SGD = SGDClassifier() # 74.19
    SGD.fit(X_train, Y_train)
    Y_pred = SGD.predict(X_test)
    acc_SGD = round(SGD.score(X_train, Y_train) * 100, 2)
    print("SGD", acc_SGD)

    svc = SVC() # 86.42
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    print("svc", acc_svc)

    linear = LinearSVC() # 79.57
    linear.fit(X_train, Y_train)
    Y_pred = linear.predict(X_test)
    acc_linear = round(linear.score(X_train, Y_train) * 100, 2)
    print("linear", acc_linear)
    """

if __name__== "__main__":
    Model(train_input, test_input)
    """
    LogisticRegression: 80.92
    DecisionTree 96.3
    RandomForest 95.4
    KNeighbors 85.86
    Gaussian 80.13
    SGD 67.9
    svc 86.08
    linear 72.95
    """