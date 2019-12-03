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

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
combine = [train_df, test_df] # reference copy

print('=== Col names ===')
print(train_df.columns.values)
print()

# preview the data
print('=== First 5 rows ===')
print(train_df.head())
print()

print('=== Last 5 rows ===')
print(train_df.tail())
print()
# train_df.info()
# print('_'*40)
# test_df.info()

# stats about the data
print('=== Null percentage ===')
print('*** train_df ***')
print(train_df.isna().sum() / train_df.shape[0])
print()
print('*** test_df ***')
print(test_df.isna().sum() / test_df.shape[0])
print()

print('=== Col stats ===')
print('*** train_df ***')
print(train_df.describe())
print()
print('*** test_df ***')
print(test_df.describe())
print()

print(train_df.describe(include=['O']))
print()

print('=== Different domain comparision ===')
# diff domain associated with survival rate
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

print()

print('=== Drop useless cols [PassengerId, Ticket, Cabin] ===')
print("Before droping") 
print("    train_df.shape: ", train_df.shape)
print("    test_df.shape: ", test_df.shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print("After droping") 
print("    train_df.shape: ", train_df.shape)
print("    test_df.shape: ", test_df.shape)

print('=== New col names ===')
print(train_df.columns.values)
print()

print('=== extract title from name ===')
# this will loop through all rows in both train and test, it updates train_df, test_df as well
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # use regex to generate Title

print('=== combine[0]: train_df ===')
print(combine[0].head())
print()
print('=== combine[1]: test_df ===')
print(combine[1].head())
print()

# cross info
# print(pd.crosstab(train_df['Title'], train_df['Sex']))

# categorize title to Miss/Mrs/Rare/Master/Mr
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Col', 'Lady', 'Countess','Capt', 'Col' 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Don'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# convert the categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    for key, val in title_mapping.items():
        dataset['Title'] = dataset['Title'].replace(key, int(val))

# Drop name
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

print(train_df.head())
print()

print('=== convert Sex to ordinal ===')
sex_mapping = {'female': 1, 'male': 0}
for dataset in combine:
    for key, val in sex_mapping.items():
        dataset['Sex'] = dataset['Sex'].replace(key, int(val))


print('=== Fill in missing age ===')
'''
More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, 
and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. 
So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
'''
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

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
            # df.loc: access a group of rows and columns by label(s) or a boolean array.
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)
print(train_df.tail())

print('=== Group age into bins ===')
# print(train_df.describe()) shows 0.0 <= Age <= 80.0
# Use cut when you need to segment and sort data values into bins, here to put age into 5 bins
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df.head())
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

print('=== convert age to ordinal value based on age bins ===')
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
print(train_df.head())
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

print('=== Null percentage (pay attention on Age) ===')
print('*** train_df ***')
print(train_df.isna().sum() / train_df.shape[0])
print()
print('*** test_df ***')
print(test_df.isna().sum() / test_df.shape[0])
print()


print('=== Fill in missing fare in test ===')
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print('*** train_df ***')
print(train_df.isna().sum() / train_df.shape[0])
print()
print('*** test_df ***')
print(test_df.isna().sum() / test_df.shape[0]) # same as print(combine[1].isna().sum() / combine[1].shape[0])
print()
print()

print('=== Group fare into bins ===')
# print(train_df.describe()) shows 0.0 <= Age <= 512.329200
# Use cut when you need to segment and sort data values into bins, here to put fare into 4 bins
train_df['FareBand'] = pd.cut(train_df['Fare'], 4)
print(train_df.tail())
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

print('=== convert fare to ordinal value based on fare bins ===')
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
print(combine[0].head())
print(combine[1].head())
print()

print('=== Fill in missing embarked in train ===')
# only 2 missing Embarked in train_df, simply replace them with the most frequent one
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
print('*** train_df ***')
print(train_df.isna().sum() / train_df.shape[0])
print()
print('*** test_df ***')
print(test_df.isna().sum() / test_df.shape[0]) # same as print(combine[1].isna().sum() / combine[1].shape[0])
print()
print()

print('=== convert Embarked to ordinal after filling in null Embarked ===')
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataset in combine:
    for key, val in embarked_mapping.items():
        dataset['Embarked'] = dataset['Embarked'].replace(key, int(val))
print(train_df.head())


print('=== Create family size ===')
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print()
print()


print('=== Create IsAlone ===')
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(combine[0].head())
print(combine[1].head())
print()

# Model
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


