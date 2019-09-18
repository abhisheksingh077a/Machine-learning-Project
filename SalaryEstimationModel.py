# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:10:33 2019

@author: Abhishek singh
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # matrix of features
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn . model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


# fitting the line in data

from sklearn . linear_model import LinearRegression

regression = LinearRegression ( )

regression . fit (X_train, y_train)

#Predicting the test to use model

y_pred = regression . predict (X_test)

#visualising the training set results

plt . scatter ( X_train, y_train, color = 'red')
plt . plot (X_train, regression . predict (X_train), color = 'blue')
plt . title ('Salary vs Experience (Training set)')
plt . xlabel ('years of experience')
plt . ylabel ('Salary')
plt . show ( )

#visualising test set results

plt . scatter ( X_test, y_test, color = 'red')
plt . plot (X_train, regression . predict (X_train), color = 'blue')
plt . title ('Salary vs Experience (Training set)')
plt . xlabel ('years of experience')
plt . ylabel ('Salary')
plt . show ( )







