import pandas as pd
import numpy as np
import random as rnd

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
combine = [train_df, test_df]

print('=== Col names ===')
print(train_df.columns.values)
print()

# preview the data
print('=== First 10 rows ===')
print(train_df.head().to_string())
print()

print('=== Last 10 rows ===')
print(train_df.tail().to_string())
print()
# train_df.info()
# print('_'*40)
# test_df.info()

# stats about the data
print('=== Null percentage ===')
print(train_df.isna().sum() / train_df.shape[0])
print()

print('=== Col stats ===')
print(train_df.describe().to_string())
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
print(train_df[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*40)
print(train_df[["Age", "Survived"]].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False).to_string())

print()
