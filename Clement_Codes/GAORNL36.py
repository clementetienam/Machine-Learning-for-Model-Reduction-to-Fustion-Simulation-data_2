# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 00:14:07 2019

@author: mjkiqce3
"""

import numpy as np

def train(c, x, y):
    """c is an array of interior breakpoints
        (not including endpoints of x)"""
    cols = [np.ones(x.shape)]
    cols.append(x)
    for i in range(len(c)):
        cols.append((x > c[i]) * (x - c[i]))
    X = np.stack(cols, axis=1) # design matrix
    alpha = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    return alpha

def predict(c, alpha, x):
    yhat = alpha[0] + alpha[1] * x
    for i in range(len(c)):
        yhat += alpha[i + 2] * (x > c[i]) * (x - c[i])
    return yhat

import matplotlib.pylab as plt
print('cluster with X and y')
X = open("inpiecewise.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(10000,2), 'F') 
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
matrix=np.concatenate((X,y), axis=1)
#
#xtest = open("intestpiecewise.out") #533051 by 28
#xtest = np.fromiter(xtest,float)
#xtest = np.reshape(xtest,(10000,1), 'F')  
#
xtest=X
inputtest=xtest

c0 = np.array([-10,-8,-6,-2,0,2,5,8,9]) # must have some points in each segment
alpha = train(c0, X, y)

plt.plot(X, y, '.')
plt.plot(X, predict(c0, alpha, X), 'r')
plt.grid()
plt.title('Piecewise Linear Regression, fixed breakpoints')
plt.show()