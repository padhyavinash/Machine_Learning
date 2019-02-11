# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\P10257441\\Desktop\\MY ML\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
###import statsmodels.formula.api as sm
###X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # this need to be added (Intercept)
###X_opt = X[:, [0, 1, 2, 3, 4, 5]]
###regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
###regressor_OLS.summary()
###
###X_opt = X[:, [0, 1, 3, 4, 5]]
###regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
###regressor_OLS.summary()
###
###X_opt = X[:, [0, 3, 4, 5]]
###regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
###regressor_OLS.summary()
###
###X_opt = X[:, [0, 3, 5]]
###regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
###regressor_OLS.summary()
###
###X_opt = X[:, [0, 3]]
###regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
###regressor_OLS.summary()
###

# Backward Elimination with p-values only

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)