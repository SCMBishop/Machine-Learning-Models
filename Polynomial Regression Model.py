# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:24:04 2019

@author: mcvet
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

lin_reg2= LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualizing Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('Predicted Salaries - Linear Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visulaizing Polynomial Regression results
X_grid = np.arange(min(X), max(X), .1)
X_grid = X_grid.reshape((len(X_grid)), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Predicted Salaries - Linear Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))