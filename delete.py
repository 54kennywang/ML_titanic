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
    print('Original X Y: ', data1_xy, '\n')

    # define x variables for original w/bin features to remove continuous variables
    data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
    data1_xy_bin = Target + data1_x_bin
    print('Bin X Y: ', data1_xy_bin, '\n')

    # define x and y variables for dummy features original
    data1_dummy = pd.get_dummies(data1[data1_x])  # having one-hot effect for categorical data, keep numeric unchanged
    data1_x_dummy = data1_dummy.columns.tolist()  # col names
    data1_xy_dummy = Target + data1_x_dummy
    print('Dummy X Y: ', data1_xy_dummy, '\n')
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
    # split training dataset into 0.6:0.3:0.1 subsets and return index of those subset, run model 10x with 60/30 split intentionally leaving out 10%
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
    return cv_split

def multi_classifier_model(data1, Target, data_val, MLA, data1_x_bin, cv_split):
    # create table to compare MLA metrics
    MLA_columns = ['MLA_Name', 'MLA_Parameters', 'MLA_Train_Accuracy_Mean', 'MLA_Test_Accuracy_Mean',
                   'MLA_Test_Accuracy_3*STD', 'MLA_Time']
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    # create table to compare MLA predictions
    MLA_predict = data1[Target]
    MLA_predict_on_data_val = pd.DataFrame(pd.np.empty((data_val.shape[0], 1)) * 0, columns=['Survived'])
    # print(MLA_predict_on_test)

    # index through MLA and save performance to table
    row_index = 0
    for alg in MLA:
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA_Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA_Parameters'] = str(alg.get_params())

        # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv=cv_split,
                                                    return_train_score=True)

        MLA_compare.loc[row_index, 'MLA_Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA_Train_Accuracy_Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA_Test_Accuracy_Mean'] = cv_results['test_score'].mean()
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA_Test_Accuracy_3*STD'] = cv_results[
                                                                    'test_score'].std() * 3  # let's know the worst that can happen!

        # save MLA predictions - see section 6 for usage
        alg.fit(data1[data1_x_bin], data1[Target])
        MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
        MLA_predict_on_data_val[MLA_name] = alg.predict(data_val[data1_x_bin])

        row_index += 1

    # print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by=['MLA_Test_Accuracy_Mean'], ascending=False, inplace=True)
    print(MLA_compare)
    print()
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
    Model = pd.DataFrame(data={'Predict': []})
    male_title = ['Master']  # survived titles

    for index, row in df.iterrows():

        # Question 1: Were you on the Titanic; majority died
        Model.loc[index, 'Predict'] = 0

        # Question 2: Are you female; majority survived
        if (df.loc[index, 'Sex'] == 'female'):
            Model.loc[index, 'Predict'] = 1

        # Question 3A Female - Class and Question 4 Embarked gain minimum information

        # Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0
        if ((df.loc[index, 'Sex'] == 'female') &
                (df.loc[index, 'Pclass'] == 3) &
                (df.loc[index, 'Embarked'] == 'S') &
                (df.loc[index, 'Fare'] > 8)

        ):
            Model.loc[index, 'Predict'] = 0

        # Question 3B Male: Title; set anything greater than .5 to 1 for majority survived
        if ((df.loc[index, 'Sex'] == 'male') &
                (df.loc[index, 'Title'] in male_title)
        ):
            Model.loc[index, 'Predict'] = 1

    return Model

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
    data_cleaner = [data1, data_val]

    # display_missing(data1) # age/carbin/embark
    # display_missing(data_val) # age/carbin/fare/embark

    data_cleaner = fill_in_missingness(data_cleaner)
    # display_missing(data1) # cabin/
    # display_missing(data_val) # cabin

    # delete the cabin feature/column and others previously stated to exclude in train dataset
    # drop_column = ['PassengerId', 'Cabin', 'Ticket']
    # data1.drop(drop_column, axis=1, inplace=True)

    data_cleaner = feature_engineering(data_cleaner)

    data1 = name_cleanUp(data1)

    data_cleaner = categorical_to_ordinal(data_cleaner)

    Target, data1_x, data1_x_calc, data1_xy, data1_x_bin, data1_xy_bin, data1_dummy, data1_x_dummy, data1_xy_dummy = define_var(data1)

    train1_x, test1_x, train1_y, test1_y, train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin, train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = cross_validation_split_1(data1, data1_x_calc, Target, data1_x_bin, data1_dummy, data1_x_dummy)

    MLA = all_classifiers()

    cv_split = cross_validation_split_2()

    MLA_columns, MLA_compare, MLA_predict, MLA_predict_on_data_val = multi_classifier_model(data1, Target, data_val, MLA, data1_x_bin, cv_split)

    del MLA_predict_on_data_val['Survived']
    MLA_predict_on_data_val['predict'] = round(MLA_predict_on_data_val.mean(axis=1)).astype(int)

    # submission(data_val['PassengerId'], MLA_predict_on_data_val['predict'], './output/achieve_99_models.csv')
    # classifier_accuracy_compare(MLA_compare)

    flip_a_coin(data1)

    # model data
    Tree_Predict = mytree(data1)
    print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(data1['Survived'], Tree_Predict) * 100)) # Accuracy classification score
    # Accuracy Summary Report with http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    # Where recall score = (true positives)/(true positive + false negative) w/1 being best:http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
    # And F1 score = weighted average of precision and recall w/1 being best: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    # print(metrics.classification_report(data1['Survived'], Tree_Predict)) # Build a text report showing the main classification metrics


if __name__== "__main__":
    Model()









