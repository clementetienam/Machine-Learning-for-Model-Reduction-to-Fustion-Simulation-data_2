# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 01:20:00 2019

@author: mjkiqce3
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
# generate data
np.random.seed(42)
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
# prepare a basis
k = 10
thresholds = np.percentile(X, np.linspace(0, 1, k+2)[1:-1]*100)
basis = np.hstack([X[:, np.newaxis],  np.maximum(0,  np.column_stack([X]*k)-thresholds)]) 
# fit a model
model = Lasso(0.03).fit(basis, y)
print(model.intercept_)
print(model.coef_.round(3))
plt.scatter(X, y)
plt.plot(X, y, color = 'b')
plt.plot(X, model.predict(basis), color='k')
plt.legend(['true', 'predicted'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('fitting segmented regression')
plt.show()