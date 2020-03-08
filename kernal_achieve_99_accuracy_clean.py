#load packages
import os
dirname = os.path.dirname(__file__)
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__))

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./input"]).decode("utf8"))

# Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8

# display missing-ness
def display_missing(df):
    for col in df.columns.tolist():
        print( '{} column missing values: {}/{} = {}'.format(col, df[col].isnull().sum(), df.shape[0], df[col].isnull().sum()/df.shape[0]))
    print('\n')

def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 12}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    plt.show()

def fill_in_missingness(data_cleaner):
    for dataset in data_cleaner:
        # complete missing age with median
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

        # complete embarked with mode
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

        # complete missing fare with median
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
    return data_cleaner

def feature_engineering(data_cleaner):
    for dataset in data_cleaner:
        # Discrete variables
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

        dataset['IsAlone'] = 1  # initialize to yes/1 is alone
        dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0  # now update to no/0 if family size is greater than 1

        # quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
        dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

        # Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
        # Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

        # Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    return data_cleaner

def name_cleanUp(data1):
    # cleanup rare title names
    # print(data1['Title'].value_counts())
    stat_min = 10  # while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
    title_names = (data1[
                       'Title'].value_counts() < stat_min)  # this will create a true false series with title name as index

    # apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
    data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    return data1

def define_var(data1):
    # define y variable aka target/outcome
    Target = ['Survived']
    # define x variables for original features aka feature selection
    data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize',
               'IsAlone']  # pretty name/values for charts
    data1_x_calc = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'SibSp', 'Parch', 'Age',
                    'Fare']  # coded for algorithm calculation
    data1_xy = Target + data1_x
    print('Original X Y: ', data1_xy)

    # define x variables for original w/bin features to remove continuous variables
    data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
    data1_xy_bin = Target + data1_x_bin
    print('Bin X Y: ', data1_xy_bin)

    # define x and y variables for dummy features original
    data1_dummy = pd.get_dummies(data1[data1_x])  # having one-hot effect for categorical data, keep numeric unchanged
    data1_x_dummy = data1_dummy.columns.tolist()  # col names
    data1_xy_dummy = Target + data1_x_dummy
    print('Dummy X Y: ', data1_xy_dummy)
    return Target, data1_x, data1_x_calc, data1_xy, data1_x_bin, data1_xy_bin, data1_dummy, data1_x_dummy, data1_xy_dummy

def cross_validation_split_1(data1, data1_x_calc, Target, data1_x_bin, data1_dummy, data1_x_dummy):
    # As mentioned previously, the test file provided is really validation data for competition submission.
    # So, we will use sklearn function to split the training data in two datasets; 75/25 split.
    # This is important, so we don't overfit our model. Meaning split train and test data with function defaults
    # random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
    train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target],
                                                                            random_state=0)
    train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin],
                                                                                            data1[Target],
                                                                                            random_state=0)
    train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(
        data1_dummy[data1_x_dummy], data1[Target], random_state=0)

    print("Data1 Shape: {}".format(data1.shape))
    print("Train1 Shape: {}".format(train1_x.shape))
    print("Test1 Shape: {}".format(test1_x.shape))
    return train1_x, test1_x, train1_y, test1_y, train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin, train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy

def cross_validation_split_2():
    # split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
    # #sklearn.model_selection.ShuffleSplit
    # note: this is an alternative to train_test_split
    # split training dataset into 0.6:0.3:0.1 subsets and return index of those subset, run model 10 times with 60/30 split intentionally leaving out 10%
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
    return cv_split

def multi_classifier_model_compare(data1, Target, data_val, MLA, data1_x_bin, cv_split):
    # create table to compare MLA metrics
    MLA_columns = ['MLA_Name', 'MLA_Parameters', 'MLA_Train_Accuracy_Mean', 'MLA_Test_Accuracy_Mean',
                   'MLA_Test_Accuracy_3*STD', 'MLA_Time']
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    # create table to compare MLA predictions
    MLA_predict = data1[Target] # train_y already known
    MLA_predict_on_data_val = pd.DataFrame(pd.np.empty((data_val.shape[0], 1)) * 0, columns=['Survived']) # test_y to be predicted

    # index through MLA and save performance to table
    row_index = 0
    for alg in MLA:
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA_Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA_Parameters'] = str(alg.get_params())

        # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        # data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
        cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv=cv_split, return_train_score=True)

        MLA_compare.loc[row_index, 'MLA_Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA_Train_Accuracy_Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA_Test_Accuracy_Mean'] = cv_results['test_score'].mean()
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA_Test_Accuracy_3*STD'] = cv_results['test_score'].std() * 3  # let's know the worst that can happen!

        # save MLA predictions - see section 6 for usage
        alg.fit(data1[data1_x_bin], data1[Target])
        MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin]) # record result of each classifier as a col for train_x
        MLA_predict_on_data_val[MLA_name] = alg.predict(data_val[data1_x_bin]) # record result of each classifier as a col for test_x

        row_index += 1

    # print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by=['MLA_Test_Accuracy_Mean'], ascending=False, inplace=True)
    print('======MLA_compare=====')
    print(MLA_compare)
    print('======MLA_predict=====')
    print(MLA_predict)
    print('======MLA_predict_on_data_val======')
    print(MLA_predict_on_data_val)
    # col_names, diff_model_comparision, predict_on_train_with_fold, predict_on_test
    return MLA_columns, MLA_compare, MLA_predict, MLA_predict_on_data_val

def all_classifiers():
    # Model Data
    MLA = [
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        # Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),

        # GLM
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),

        # Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        # Nearest Neighbor
        neighbors.KNeighborsClassifier(),  # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        # Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        XGBClassifier()
    ]
    return MLA

def categorical_to_ordinal(data_cleaner):
    # code categorical data
    label = LabelEncoder()  # generate number values for categorical data
    for dataset in data_cleaner:
        dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
        dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
        dataset['Title_Code'] = label.fit_transform(dataset['Title'])
        dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
        dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
    return data_cleaner

def submission(PassengerId, Survived, out):
    sub = pd.DataFrame({'PassengerId': PassengerId, 'Survived': Survived})
    outFile = os.path.join(dirname, out)
    print('===output pred to ' + out + '===')
    sub.to_csv(outFile, index=False)

def classifier_accuracy_compare(MLA_compare):
    # barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
    sns.barplot(x='MLA_Test_Accuracy_Mean', y='MLA_Name', data=MLA_compare, color='m')
    # prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
    plt.title('Machine Learning Algorithm Accuracy Score \n')
    plt.xlabel('Accuracy Score (%)')
    plt.ylabel('Algorithm')
    plt.show()

def flip_a_coin(data1):
    # IMPORTANT: This is a handmade model for learning purposes only.
    # However, it is possible to create your own predictive model without a fancy algorithm :)
    # coin flip model with random 1/survived 0/died
    # iterate over dataFrame rows as (index, Series) pairs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html
    for index, row in data1.iterrows():
        # random number generator: https://docs.python.org/2/library/random.html
        if random.random() > .5:  # Random float x, 0.0 <= x < 1.0
            data1.set_value(index, 'Random_Predict', 1)  # predict survived/1
        else:
            data1.set_value(index, 'Random_Predict', 0)  # predict died/0

    # score random guess of survival. Use shortcut 1 = Right Guess and 0 = Wrong Guess
    # the mean of the column will then equal the accuracy
    data1['Random_Score'] = 0  # assume prediction wrong
    data1.loc[(data1['Survived'] == data1['Random_Predict']), 'Random_Score'] = 1  # set to 1 for correct prediction
    print('Coin Flip Model Accuracy: {:.2f}%'.format(data1['Random_Score'].mean() * 100))

    # we can also use scikit's accuracy_score function to save us a few lines of code
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    print('Coin Flip Model Accuracy w/SciKit: {:.2f}%'.format(
        metrics.accuracy_score(data1['Survived'], data1['Random_Predict']) * 100))
    print()

# handmade data model using brain power (and Microsoft Excel Pivot Tables for quick calculations)
def mytree(df):
    # initialize table to store predictions
    myModel = pd.DataFrame(data={'Predict': []})
    male_title = ['Master']  # survived titles

    for index, row in df.iterrows():

        # Question 1: Were you on the Titanic; majority died
        myModel.loc[index, 'Predict'] = 0

        # Question 2: Are you female; majority survived
        if (df.loc[index, 'Sex'] == 'female'):
            myModel.loc[index, 'Predict'] = 1

        # Question 3A Female - Class and Question 4 Embarked gain minimum information

        # Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0
        if ((df.loc[index, 'Sex'] == 'female') &
                (df.loc[index, 'Pclass'] == 3) &
                (df.loc[index, 'Embarked'] == 'S') &
                (df.loc[index, 'Fare'] > 8)

        ):
            myModel.loc[index, 'Predict'] = 0

        # Question 3B Male: Title; set anything greater than .5 to 1 for majority survived
        if ((df.loc[index, 'Sex'] == 'male') &
                (df.loc[index, 'Title'] in male_title)
        ):
            myModel.loc[index, 'Predict'] = 1
    return myModel


def decisionTree_no_hyperparam_tune(data1, data1_x_bin, Target, cv_split):
    # hyperparam tuning
    # base model
    dtree = tree.DecisionTreeClassifier(random_state=0)
    base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv=cv_split, return_train_score=True)
    dtree.fit(data1[data1_x_bin], data1[Target])

    print('BEFORE DT Parameters: ', dtree.get_params())
    print("BEFORE DT Training w/bin score mean: {:.2f}".format(base_results['train_score'].mean() * 100))
    print("BEFORE DT Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
    print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))
    # print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))
    print('-' * 10)
    return dtree, base_results

def decisionTree_with_hyperparam_tune(data1, data1_x_bin, Target, cv_split):
    # tune hyper-parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    param_grid = {'criterion': ['gini', 'entropy'],
                  # scoring methodology; two supported formulas for calculating information gain - default is gini
                  # 'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
                  'max_depth': [2, 4, 6, 8, 10, None],  # max depth tree can grow; default is none
                  # 'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
                  # 'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
                  # 'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
                  'random_state': [0]
                  # seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
                  }

    # print(list(model_selection.ParameterGrid(param_grid)))

    # choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    # Exhaustive search over specified parameter values for an estimator
    tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
    tune_model.fit(data1[data1_x_bin], data1[Target])

    # print(tune_model.cv_results_.keys())
    # print(tune_model.cv_results_['params'])
    print('AFTER DT Parameters: ', tune_model.best_params_)
    # print(tune_model.cv_results_['mean_train_score'])
    print("AFTER DT Training w/bin score mean: {:.2f}".format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
    # print(tune_model.cv_results_['mean_test_score'])
    print("AFTER DT Test w/bin score mean: {:.2f}".format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
    print("AFTER DT Test w/bin score 3*std: +/- {:.2f}".format(tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
    print('-' * 10)
    return tune_model, param_grid

def feature_select(data1, data1_x_bin, dtree, base_results, cv_split, Target):
    # Feature Selection
    # more predictor variables do not make a better model, but the right predictors do.
    print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape)
    print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values.tolist())

    print("BEFORE DT RFE Training w/bin score mean: {:.2f}".format(base_results['train_score'].mean() * 100))
    print("BEFORE DT RFE Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
    print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))
    print('-' * 10)

    # feature selection
    dtree_rfe = feature_selection.RFECV(dtree, step=1, scoring='accuracy', cv=cv_split, n_jobs=-1)
    dtree_rfe.fit(data1[data1_x_bin], data1[Target])

    # transform x&y to reduced features and fit new model
    # alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()].tolist() # Get a mask, or integer index, of the features selected
    rfe_results = model_selection.cross_validate(dtree, data1[X_rfe], data1[Target], cv=cv_split, return_train_score=True)

    # print(dtree_rfe.grid_scores_)
    print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape)
    print('AFTER DT RFE Training Columns New: ', X_rfe)

    print("AFTER DT RFE Training w/bin score mean: {:.2f}".format(rfe_results['train_score'].mean() * 100))
    print("AFTER DT RFE Test w/bin score mean: {:.2f}".format(rfe_results['test_score'].mean() * 100))
    print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}".format(rfe_results['test_score'].std() * 100 * 3))
    print('-' * 10)
    return dtree_rfe, X_rfe, rfe_results

def multi_classifier_voting_predication(data1, data1_x_bin, cv_split, Target):
    # why choose one model, when you can pick them all with voting classifier
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    # removed models w/o attribute 'predict_proba' required for vote classifier and models with a 1.0 correlation to another model
    vote_est = [
        # Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
        ('ada', ensemble.AdaBoostClassifier()),
        ('bc', ensemble.BaggingClassifier()),
        ('etc', ensemble.ExtraTreesClassifier()),
        ('gbc', ensemble.GradientBoostingClassifier()),
        ('rfc', ensemble.RandomForestClassifier()),
        # Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
        ('gpc', gaussian_process.GaussianProcessClassifier()),

        # GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        ('lr', linear_model.LogisticRegressionCV()),

        # Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
        ('bnb', naive_bayes.BernoulliNB()),
        ('gnb', naive_bayes.GaussianNB()),

        # Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
        ('knn', neighbors.KNeighborsClassifier()),

        # SVM: http://scikit-learn.org/stable/modules/svm.html
        ('svc', svm.SVC(probability=True)),

        # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        ('xgb', XGBClassifier())

    ]
    # Hard Vote or majority rules
    vote_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
    vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target], cv=cv_split, return_train_score=True)
    vote_hard.fit(data1[data1_x_bin], data1[Target])

    print("Hard Voting Training w/bin score mean: {:.2f}".format(vote_hard_cv['train_score'].mean() * 100))
    print("Hard Voting Test w/bin score mean: {:.2f}".format(vote_hard_cv['test_score'].mean() * 100))
    print("Hard Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_hard_cv['test_score'].std() * 100 * 3))
    print('-' * 10)

    # Soft Vote or weighted probabilities
    vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
    vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], cv=cv_split,
                                                  return_train_score=True)
    vote_soft.fit(data1[data1_x_bin], data1[Target])

    print("Soft Voting Training w/bin score mean: {:.2f}".format(vote_soft_cv['train_score'].mean() * 100))
    print("Soft Voting Test w/bin score mean: {:.2f}".format(vote_soft_cv['test_score'].mean() * 100))
    print("Soft Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_soft_cv['test_score'].std() * 100 * 3))
    print('-' * 10)
    return vote_hard, vote_soft

def draw_tree(my_tree, feature_names, out='output/dtree_render'):
    # Graph MLA version of Decision Tree: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
    import graphviz
    dot_data = tree.export_graphviz(my_tree, out_file=None, feature_names=feature_names, class_names=True, filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render(out, view=True)

def Model():
    data_raw = pd.read_csv('./input/train.csv')

    # a dataset should be broken into 3 splits: train, test, and (final) validation
    # the test file provided is the validation file for competition submission
    # we will split the train set into train and test data in future sections
    data_val = pd.read_csv('./input/test.csv')

    # to play with our data we'll create a copy
    # remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
    data1 = data_raw.copy(deep=True)

    # however passing by reference is convenient, because we can clean both datasets at once
    data_cleaner = [data1, data_val] # [train_copy, test]

    display_missing(data1) # age/carbin/embark
    display_missing(data_val) # age/carbin/fare/embark

    data_cleaner = fill_in_missingness(data_cleaner) # age/fare/embark
    # display_missing(data1) # cabin
    # display_missing(data_val) # cabin

    # delete the cabin feature/column and others previously stated to exclude in train dataset
    # drop_column = ['PassengerId', 'Cabin', 'Ticket']
    # data1.drop(drop_column, axis=1, inplace=True)

    data_cleaner = feature_engineering(data_cleaner) # create FamilySize/IsAlone/Title/FareBin/AgeBin

    data1 = name_cleanUp(data1) # create Title

    data_cleaner = [data1, data_val]  # [train_copy, test]


    # print('kenny') # this code is for data_cleaner = categorical_to_ordinal(data_cleaner)
    # print(data_cleaner[0].head(1)) # this code is for data_cleaner = categorical_to_ordinal(data_cleaner)
    # print(data_cleaner[1].head(1)) # this code is for data_cleaner = categorical_to_ordinal(data_cleaner)
    data_cleaner = categorical_to_ordinal(data_cleaner)
    # print(data_cleaner[0].head(1)) # this code is for data_cleaner = categorical_to_ordinal(data_cleaner)
    # print(data_cleaner[1].head(1)) # this code is for data_cleaner = categorical_to_ordinal(data_cleaner)
    # print('kenny')

    # only data1_dummy is df, all others are [col names]
    Target, data1_x, data1_x_calc, data1_xy, data1_x_bin, data1_xy_bin, data1_dummy, data1_x_dummy, data1_xy_dummy = define_var(data1)

    # do not understand why we split data here
    train1_x, test1_x, train1_y, test1_y, train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin, train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = cross_validation_split_1(data1, data1_x_calc, Target, data1_x_bin, data1_dummy, data1_x_dummy)
    MLA = all_classifiers()

    cv_split = cross_validation_split_2()

    # col_names, diff_model_comparision, predict_on_train_with_fold, predict_on_test
    MLA_columns, MLA_compare, MLA_predict, MLA_predict_on_data_val = multi_classifier_model_compare(data1, Target, data_val, MLA, data1_x_bin, cv_split)
    del MLA_predict_on_data_val['Survived'] # remove the useless col that was initially created for num of rows
    MLA_predict_on_data_val['predict'] = round(MLA_predict_on_data_val.mean(axis=1)).astype(int) # let final prediction be avg among all models' prediction

    # submission(data_val['PassengerId'], MLA_predict_on_data_val['predict'], './output/achieve_99_models.csv') # 0.77511
    # classifier_accuracy_compare(MLA_compare)

    flip_a_coin(data1)

    # model data
    Tree_Predict = mytree(data1)
    print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(data1['Survived'], Tree_Predict) * 100)) # Accuracy classification score
    Tree_Predict_submit = mytree(data_val).astype(int)
    # submission(data_val['PassengerId'], Tree_Predict_submit['Predict'], './output/achieve_99_mytree.csv') # 0.77990

    # support is the number of samples of the true response that lie in that class.
    # macro average (averaging the unweighted mean per label)
    print('kenny')
    print(metrics.classification_report(data1['Survived'], Tree_Predict)) # Build a text report showing the main classification metrics
    # hyper-param tuning
    dtree, base_results = decisionTree_no_hyperparam_tune(data1, data1_x_bin, Target, cv_split)
    y_pred_dtree = dtree.predict(data_val[data1_x_bin]).astype(int)
    # submission(data_val['PassengerId'], y_pred_dtree, './output/achieve_99_dtree.csv') # 0.76555

    tune_model, param_grid = decisionTree_with_hyperparam_tune(data1, data1_x_bin, Target, cv_split)
    y_pred_tune_model = tune_model.predict(data_val[data1_x_bin]).astype(int)
    submission(data_val['PassengerId'], y_pred_tune_model, './output/achieve_99_tune_model.csv')

    # feature selection
    dtree_rfe, X_rfe, rfe_results = feature_select(data1, data1_x_bin, dtree, base_results, cv_split, Target)
    y_pred_dtree_rfe = dtree_rfe.predict(data_val[data1_x_bin]).astype(int)
    submission(data_val['PassengerId'], y_pred_dtree_rfe, './output/achieve_99_dtree_rfe.csv')

    # draw_tree(dtree, data1_x_bin, out='output/dtree')

    vote_hard, vote_soft = multi_classifier_voting_predication(data1, data1_x_bin, cv_split, Target)
    y_pred_vote_hard = vote_hard.predict(data_val[data1_x_bin]).astype(int)
    submission(data_val['PassengerId'], y_pred_vote_hard, './output/achieve_99_vote_hard.csv')

    y_pred_vote_soft = vote_soft.predict(data_val[data1_x_bin]).astype(int)
    submission(data_val['PassengerId'], y_pred_vote_soft, './output/achieve_99_vote_soft.csv')

if __name__== "__main__":
    Model()









