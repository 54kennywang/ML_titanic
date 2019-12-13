import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import numpy as np


d1 = {'Sex': ['male', 'female']}
d1 = pd.DataFrame(data=d1)
print(d1)
label = LabelEncoder()
d1['Sex_Code'] = label.fit_transform(d1['Sex'])
print(d1)

data1_x = ['Sex']
data1_x_calc = ['Sex_Code']
Target = ['Survived']
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')

data1_x_bin = ['Sex_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')

data1_dummy = pd.get_dummies(d1[data1_x])
print(data1_dummy, '\n')
data1_x_dummy = data1_dummy.columns.tolist()
print(data1_x_dummy, '\n')
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')