import pandas as pd
import numpy as np
import random as rnd

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
train_df = train_df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
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
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col' 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
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


''' missing data in Embarked
print('=== convert Embarked to ordinal ===')
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataset in combine:
    for key, val in embarked_mapping.items():
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)
print(train_df.head())
'''

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











