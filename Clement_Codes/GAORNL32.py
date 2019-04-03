# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 23:44:17 2019

@author: mjkiqce3
"""

from patsy import dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 


X = open("inpiecewise.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(10000,2), 'F') 
#X=X[:,0]
#X=np.reshape(X,(-1,1))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
(scaler.fit(X))
X=(scaler.transform(X))
y = open("outpiecewise.out") #533051 by 28
y = np.fromiter(y,float)
y = np.reshape(y,(10000,1), 'F')  

ydami=y

#ytest = open("outtestpiecewise.out") #533051 by 28
#ytest = np.fromiter(ytest,float)
#ytest = np.reshape(ytest,(10000,1), 'F')  

ytest=y
outputtest=ytest

numrowstest=len(outputtest)

inputtrainclass=X
#
outputtrainclass=y
xtest=X
inputtest=xtest
matrix=np.concatenate((X,y), axis=1)
train_x=X
train_y=y
valid_x=X
valid_y=y
# Generating cubic spline with 3 knots at 25, 40 and 60
transformed_x = dmatrix("bs(train, knots=(-8,-7,-2,0,3,5,8), degree=7, include_intercept=False)", {"train": train_x})

# Fitting Generalised linear model on transformed dataset
fit1 = sm.GLM(train_y, transformed_x).fit()

# Generating cubic spline with 4 knots
transformed_x2 = dmatrix("bs(train, knots=(25,40,50,65),degree =3, include_intercept=False)", {"train": train_x}, return_type='dataframe')

# Fitting Generalised linear model on transformed dataset
fit2 = sm.GLM(train_y, transformed_x2).fit()

# Predictions on both splines
pred1 = fit1.predict(dmatrix("bs(valid, knots=(25,40,60), include_intercept=False)", {"valid": valid_x}, return_type='dataframe'))
pred2 = fit2.predict(dmatrix("bs(valid, knots=(25,40,50,65),degree =3, include_intercept=False)", {"valid": valid_x}, return_type='dataframe'))

# Calculating RMSE values
rms1 = sqrt(mean_squared_error(valid_y, pred1))

rms2 = sqrt(mean_squared_error(valid_y, pred2))

# We will plot the graph for 70 observations only
xp = np.linspace(valid_x.min(),valid_x.max(),70)

# Make some predictions
pred1 = fit1.predict(dmatrix("bs(xp, knots=(25,40,60), include_intercept=False)", {"xp": xp}, return_type='dataframe'))
pred2 = fit2.predict(dmatrix("bs(xp, knots=(25,40,50,65),degree =3, include_intercept=False)", {"xp": xp}, return_type='dataframe'))

# Plot the splines and error bands
plt.scatter(data.age, data.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(xp, pred1, label='Specifying degree =3 with 3 knots')
plt.plot(xp, pred2, color='r', label='Specifying degree =3 with 4 knots')
plt.legend()
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
plt.show()