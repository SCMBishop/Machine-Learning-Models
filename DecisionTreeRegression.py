# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:17:57 2019

@author: mcvet
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

dataset['Salary']=dataset.Salary.astype(float)
dataset['Level']=dataset.Level.astype(float)
dataset.dtypes

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

'''
#Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state =0)
regressor.fit(X, y)


# Predicting a new result
y_pred = regressor.predict([[6.5]])
#y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
#prediction = sc_y.inverse_transform(y_pred)

# Visualising the Regression results
#non-linear and non-continuous so it must be graphed in high ressolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decission Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()