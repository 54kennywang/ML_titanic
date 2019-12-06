# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook

"""
stacking: we have x_train -> y_train, we want to predict x_test -> ?
normally, we train a stack m1 by (x_train, y_train), then we use m1 to do x_test -> y_test, y_test is our final predication --- (1)
here, in (1), when m1 is being trained, we use oof to get y_train_oof on x_train, and ofc we still get y_test
in second stack m2, we train m2 by (y_train_oof, y_train), then we use m2 to do y_test -> final_predication on x_test

oof: when train a model m by (x, y), instead of using all x at once to get loss by y, we divide x to (x1, x2, x3) (assume 3 fold)
     x1, x2, x3 are disjoint and x1 + x2 + x3 = x
     1) we train m by ((x1, x2), (y1, y2)) and we predict y3_oof on x3 and y_test_1 on y
     2) we train m by ((x2, x3), (y2, y3)) and we predict y1_oof on x1 and y_test_2 on y
     3) we train m by ((x1, x3), (y1, y3)) and we predict y2_oof on x2 and y_test_3 on y
     finally, y1_oof + y2_oof + y3_oof = y_train_oof, average (y_test_1, y_test_2, y_test_3) to be y_test
     so now, for layer 2, we have (y_train_oof, y) to train m2, then use m2 to predict (y_test, final_predication)
"""

# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import os
dirname = os.path.dirname(__file__)

# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

# set dataFrame display properties
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load in the train and test datasets
train_input = os.path.join(dirname, './input/train.csv')
test_input = os.path.join(dirname, './input/test.csv')
train = pd.read_csv(train_input)
test = pd.read_csv(test_input)
PassengerId = test['PassengerId']

# Some useful parameters which will come in handy later on
ntrain = train.shape[0] # (891, 12)
ntest = test.shape[0] # (418, 11)
SEED = 0  # for reproducibility
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state = SEED)

# params for different models
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# all feature engineering before training
# input raw train and test dataFrame from csv
# output engineered np.array
def fist_layer_feature_engineering(train, test):
    # Store our passenger ID for easy access
    PassengerId = test['PassengerId']
    full_data = [train, test]

    # Gives the length of the name
    train['Name_length'] = train['Name'].apply(len)
    test['Name_length'] = test['Name'].apply(len)

    # Feature that tells whether a passenger had a cabin on the Titanic
    train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Create new feature FamilySize as a combination of SibSp and Parch
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

    # Create a New feature CategoricalAge
    for dataset in full_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    train['CategoricalAge'] = pd.cut(train['Age'], 5)

    # Create a new feature Title, containing the titles of passenger names
    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    for dataset in full_data:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # Mapping Fare
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # Mapping Age
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4;

    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train = train.drop(drop_elements, axis=1)
    train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
    test = test.drop(drop_elements, axis=1)

    # Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
    y_train = train['Survived'].ravel()
    train = train.drop(['Survived'], axis=1)
    tr = train
    x_train = train.values  # Creates an array of the train data
    x_test = test.values  # Creats an array of the test data
    return x_train, y_train, x_test, tr

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return (self.clf.fit(x, y).feature_importances_)

# Out-of-Fold Predictions
# one cannot simply train the base models on the full training data, generate predictions on the full test set and then output these for the second-level training. This runs the risk of your base model predictions already having "seen" the test set and therefore overfitting when feeding these predictions.
# clf is the model
# x_train, y_train, x_test are the outputs from fist_layer_feature_engineering
# return prediction on x_train, average predication on x_test
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,)) # prediction on x_train
    oof_test = np.zeros((ntest,)) # average of 5 times of predication on x_test
    oof_test_skf = np.empty((NFOLDS, ntest)) # each row is the predication for x_test

    for i, (train_index, test_index) in enumerate(kf.split(x_train)): # i is the time of looping 0-4
        x_tr = x_train[train_index] # train subset
        y_tr = y_train[train_index] # train label subset
        x_te = x_train[test_index] # remaining train subset as test

        clf.train(x_tr, y_tr) # train on k-1 folds of train
        oof_train[test_index] = clf.predict(x_te) # predict on last fold of train

        oof_test_skf[i, :] = clf.predict(x_test) # single time prediction on test

    oof_test[:] = oof_test_skf.mean(axis=0) # average of 5 times of predication

    # return prediction on train, average predication on test
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# create all models for first layer, return a list of models
def create_first_layer_models():
    rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
    et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
    gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
    svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
    return [rf, et, ada, gb, svc]

# train first layer model with feature-engineered data
# return oof_train: the tuple of train predictions for all models using oof
# return oof_test: the tuple of test predictions for all models using oof
def fist_layer_training(models, x_train, y_train, x_test):
    rf, et, ada, gb, svc = models
    et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
    ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
    gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
    svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier
    oof_train = (et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train)
    oof_test = (et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test)
    return oof_train, oof_test

# concatenated and joined both the first-level train and test predictions as x_train and x_test, we can now fit a second-level learning model.
def produce_second_input_from_first_output(oof_train, oof_test):
    x_train_2 = np.concatenate(oof_train, axis=1)
    x_test_2 = np.concatenate(oof_test, axis=1)
    return x_train_2, x_test_2

def create_second_layer_model(x_train_2, x_test_2):
    gbm = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=4,
        min_child_weight=2,
        # gamma=1,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        # learning_rate = 0.02,
        scale_pos_weight=1).fit(x_train_2, x_test_2)
    return gbm

def submission(PassengerId, predictions):
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId,
                                       'Survived': predictions})
    outFile = os.path.join(dirname, './output/StackingSubmission.csv')
    StackingSubmission.to_csv(outFile, index=False)

"""
Pearson Correlation of Features
There are not too many features strongly correlated with one another. This is good from a point of view of feeding these features into your learning model because this means that there isn't much redundant or superfluous data in our training set and we are happy that each feature carries with it some unique information.
"""
def Pearson_Correlation_of_Features(train):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()

def feature_importance(x_train, y_train, models, tr):
    rf = models[0]
    et = models[1]
    ada = models[2]
    gb = models[3]
    rf_feature = rf.feature_importances(x_train, y_train).tolist()
    et_feature = et.feature_importances(x_train, y_train).tolist()
    ada_feature = ada.feature_importances(x_train, y_train).tolist()
    gb_feature = gb.feature_importances(x_train, y_train).tolist()

    titles = (tr.columns.values.tolist())
    Models = ['rf', 'et', 'ada', 'gb']
    df = pd.DataFrame([rf_feature, et_feature, ada_feature, gb_feature], index=Models, columns=titles)
    df.loc['mean'] = df.mean()
    df.to_csv('feature_importance.csv')

def Model(train, test):
    # feature engineer the first stack
    x_train, y_train, x_test, tr = fist_layer_feature_engineering(train, test)

    # get Pearson_Correlation_of_Features
    # x_train_df = pd.DataFrame(data=x_train, columns=
    # ['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Name_length', 'Has_Cabin',
    #  'FamilySize', 'IsAlone', 'Title'])
    # Pearson_Correlation_of_Features(x_train_df)

    # create all models for first stack training
    models = create_first_layer_models()

    # train first stack
    oof_train, oof_test = fist_layer_training(models, x_train, y_train, x_test)
    print('===first layer training finished===')

    # option to output feature importance and use excel pie chart to visualize
    # feature_importance(x_train, y_train, models, tr)

    # produce second stack input from first stack output
    x_train_2, x_test_2 = produce_second_input_from_first_output(oof_train, oof_test)

    # train second stack
    gbm = create_second_layer_model(x_train_2, y_train)

    # predict out final prediction
    predictions = gbm.predict(x_test_2)

    acc_gbm = round(gbm.score(x_train_2, y_train) * 100, 2)
    print('Train accuracy:', acc_gbm)
    # submission(PassengerId, predictions)

if __name__== "__main__":
    Model(train, test)
