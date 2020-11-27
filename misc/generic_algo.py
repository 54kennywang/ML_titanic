"""
This code is very incomplete, see the Generic Algorithm note for future investigation
"""

# imports
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler,MinMaxScaler
import pickle
import string
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import pandas as pd
from sympy import simplify, cos, sin, Symbol, Function, tanh, pprint, init_printing, exp
from sympy.functions import Min,Max

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# now this number indicates the number of generations, which can quite long.
# I recommend something around the number of 1000, but 100 is timewise okay (~5 min)
# Currently it is at 10 to reduce runtime. Of course the more iterations the better the algorithm
# Some code taken from: https://github.com/innjoshka/Genetic-Programming-Titanic-Kaggle
HOWMANYITERS = 10


def GP_deap(evolved_train):
    global HOWMANYITERS
    import operator
    import math
    import random

    from deap import algorithms
    from deap import base, creator
    from deap import tools
    from deap import gp

    # dropping Survived and Passenger ID because we can not use them for training
    outputs = evolved_train['Survived'].values.tolist()
    evolved_train = evolved_train.drop(["Survived", "PassengerId"], axis=1)
    inputs = evolved_train.values.tolist()  # to np array

    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    # generate a random string of length 10
    def randomString(stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))

    # choosing Primitives
    pset = gp.PrimitiveSet("MAIN", len(evolved_train.columns))  # add here
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.tanh, 1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2)
    pset.addEphemeralConstant(randomString(), lambda: random.uniform(-10, 10))
    # 50 as a precaution. 34 would be enough
    pset.renameArguments(ARG0='x1')
    pset.renameArguments(ARG1='x2')
    pset.renameArguments(ARG2='x3')
    pset.renameArguments(ARG3='x4')
    pset.renameArguments(ARG4='x5')
    pset.renameArguments(ARG5='x6')
    pset.renameArguments(ARG6='x7')
    pset.renameArguments(ARG7='x8')
    pset.renameArguments(ARG8='x9')
    pset.renameArguments(ARG9='x10')
    pset.renameArguments(ARG10='x11')
    pset.renameArguments(ARG11='x12')
    pset.renameArguments(ARG12='x13')
    pset.renameArguments(ARG13='x14')
    pset.renameArguments(ARG14='x15')
    pset.renameArguments(ARG15='x16')
    pset.renameArguments(ARG16='x17')
    pset.renameArguments(ARG17='x18')
    pset.renameArguments(ARG18='x19')
    pset.renameArguments(ARG19='x20')
    pset.renameArguments(ARG20='x21')
    pset.renameArguments(ARG21='x22')
    pset.renameArguments(ARG22='x23')
    pset.renameArguments(ARG23='x24')
    pset.renameArguments(ARG24='x25')
    pset.renameArguments(ARG25='x26')
    pset.renameArguments(ARG26='x27')
    pset.renameArguments(ARG27='x28')
    pset.renameArguments(ARG28='x29')
    pset.renameArguments(ARG29='x30')
    pset.renameArguments(ARG30='x31')
    pset.renameArguments(ARG31='x32')
    pset.renameArguments(ARG32='x33')
    pset.renameArguments(ARG33='x34')
    pset.renameArguments(ARG34='x35')
    pset.renameArguments(ARG35='x36')
    pset.renameArguments(ARG36='x37')
    pset.renameArguments(ARG37='x38')
    pset.renameArguments(ARG38='x39')
    pset.renameArguments(ARG39='x40')
    pset.renameArguments(ARG40='x41')
    pset.renameArguments(ARG41='x42')
    pset.renameArguments(ARG42='x43')
    pset.renameArguments(ARG43='x44')
    pset.renameArguments(ARG44='x45')
    pset.renameArguments(ARG45='x46')
    pset.renameArguments(ARG46='x47')
    pset.renameArguments(ARG47='x48')
    pset.renameArguments(ARG48='x49')
    pset.renameArguments(ARG49='x50')

    # two object types is needed: an individual containing the genotype
    # and a fitness -  The reproductive success of a genotype (a measure of quality of a solution)
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # register some parameters specific to the evolution process.
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)  #
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # evaluation function, which will receive an individual as input, and return the corresponding fitness.
    def evalSymbReg(individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the accuracy of individuals // 1|0 == survived
        return math.fsum(
            np.round(1. - (1. / (1. + np.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs)) / len(
            evolved_train),

    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    # Statistics over the individuals fitness and size
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=HOWMANYITERS, stats=stats,
                                   halloffame=hof, verbose=True)

    # Parameters:
    # population – A list of individuals.
    # toolbox – A Toolbox that contains the evolution operators.
    # cxpb – The probability of mating two individuals.
    # mutpb – The probability of mutating an individual.
    # ngen – The number of generation.
    # stats – A Statistics object that is updated inplace, optional.
    # halloffame – A HallOfFame object that will contain the best individuals, optional.
    # verbose – Whether or not to log the statistics.

    # Transform the tree expression of hof[0] in a callable function and return it
    func2 = toolbox.compile(expr=hof[0])

    return func2


def manualtree(df):
    # using manualtree from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
    # initialize table to store predictions
    Model = pd.DataFrame(data={'manual_tree': []})
    male_title = ['Master']  # survived titles

    for index, row in df.iterrows():

        # Question 1: Were you on the Titanic; majority died
        Model.loc[index, 'manual_tree'] = 0

        # Question 2: Are you female; majority survived
        if (df.loc[index, 'Sex'] == 'female'):
            Model.loc[index, 'manual_tree'] = 1

        # Question 3A Female - Class and Question 4 Embarked gain minimum information

        # Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0
        if ((df.loc[index, 'Sex'] == 'female') &
                (df.loc[index, 'Pclass'] == 3) &
                (df.loc[index, 'Embarked'] == 'S') &
                (df.loc[index, 'Fare'] > 8)

        ):
            Model.loc[index, 'manual_tree'] = 0

        # Question 3B Male: Title; set anything greater than .5 to 1 for majority survived
        if ((df.loc[index, 'Sex'] == 'male') &
                (df.loc[index, 'Title'] == 3)
        ):
            Model.loc[index, 'manual_tree'] = 1

    return Model


def MungeData(data):
    title_list = [
        'Dr', 'Mr', 'Master',
        'Miss', 'Major', 'Rev',
        'Mrs', 'Ms', 'Mlle', 'Col',
        'Capt', 'Mme', 'Countess',
        'Don', 'Jonkheer'
    ]

    # replacing all people's name by their titles
    def replace_names_titles(x):
        for title in title_list:
            if title in x:
                return title

    data['Title'] = data.Name.apply(replace_names_titles)
    data['Title'] = data['Title'].replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
                                          'Miss/Mrs/Ms')
    data['Title'] = data['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
                                          'Dr/Military/Noble/Clergy')
    data['Title'] = data.Title.map(
        {'Dr': 1, 'Mr': 2, 'Master': 3, 'Miss': 4, 'Major': 5, 'Rev': 6, 'Mrs': 7, 'Ms': 8, 'Mlle': 9,
         'Col': 10, 'Capt': 11, 'Mme': 12, 'Countess': 13, 'Don': 14, 'Jonkheer': 15
         })
    data = data.drop(['Name'], axis=1)
    data.Title.fillna(0, inplace=True)
    data['Is_Married'] = 0
    data['Is_Married'].loc[data['Title'] == 7] = 1
    # manual_tree
    data["manual_tree"] = manualtree(data)
    # Age
    data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    # Relatives
    data['Relatives'] = data.SibSp + data.Parch
    # Fare per person
    data['Fare_per_person'] = data.Fare / np.mean(data.SibSp + data.Parch + 1)
    # data.drop(['Fare'], inplace=True, axis=1)
    med_fare = data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    data = data.drop(['SibSp', 'Parch'], axis=1)
    # Filling the missing value in Fare with the median Fare of 3rd class alone passenger
    data['Fare'] = data['Fare'].fillna(med_fare)
    # Ticket
    # Sex
    data.Sex.fillna('0', inplace=True)
    data.loc[data.Sex != 'male', 'Sex'] = 0
    data.loc[data.Sex == 'male', 'Sex'] = 1
    data['Ticket_Frequency'] = data.groupby('Ticket')['Ticket'].transform('count')
    data = data.drop(['Ticket'], axis=1)
    # Cabin
    data.Cabin.fillna('0', inplace=True)
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 3
    # Embarked
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2
    data.loc[data.Embarked == 'S', 'Embarked'] = 3
    data.Embarked.fillna(3, inplace=True)
    # data.fillna(0, inplace=True)
    # print(data.columns)#data["Survived"] = svd_tmp
    data["Cabin"] = data["Cabin"].astype(int)
    # now for encoding - first we scale numeric features. E.g. Fare will have bigger values as Age, which
    # could confuse an algorithm. therefore we normalize values in the range (-1,1)
    numeric_features = ['Relatives', 'Fare_per_person', 'Fare', 'Age', 'Ticket_Frequency']
    for feature in numeric_features:
        x = data[feature].values  # returns a numpy array
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
        data[feature] = pd.DataFrame(x_scaled)

    # Categorial features
    # Now the best thing for algorithms to work with categories is to have the category values in different
    # columns as either 1 or 0.
    cat_features = ['Pclass', 'Embarked', 'Sex', 'Cabin', 'Title', 'manual_tree', 'Is_Married']
    encoded_features = []
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(data[feature].values.reshape(-1, 1)).toarray()
        n = data[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = data.index
        encoded_features.append(encoded_df)
    data = pd.concat([data, *encoded_features], axis=1)
    return data.astype(float)
