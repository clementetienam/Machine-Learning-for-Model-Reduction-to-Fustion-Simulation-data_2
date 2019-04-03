# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:13:17 2019

@author: Dr Clement Etienam
"""


print(__doc__)
import numpy as np
from piecewise.regressor import piecewise
import matplotlib.pyplot as plt


X = open("inputactive.out") #533051 by 28
X = np.fromiter(X,float)
#X = np.reshape(X,(1000,1), 'F')  


y = open("outputactive.out") #533051 by 28
y = np.fromiter(y,float)


ytest = open("outputtestactive.out") #533051 by 28
ytest = np.fromiter(ytest,float)


xtest = open("inputtestactive.out") #533051 by 28
xtest = np.fromiter(xtest,float)

model = piecewise(X, y)

a=len(model.segments)

ypredict=model.predict(xtest)

fig = plt.figure()
plt.plot( xtest, ytest, 'ok' );

plt.plot( xtest,ypredict,'or' );
plt.xlabel('X'); plt.ylabel('Y');


fig = plt.figure()
plt.scatter(ypredict,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('GP estimate')
plt.title('Machine  prediction on discountionus function')

dd=ytest-ypredict

from piecewise.plotter import plot_data_with_regression

plot_data_with_regression(X, y)


fig = plt.figure()
plt.plot(ytest, color = 'red', label = 'Real data')
plt.plot(ypredict, color = 'blue', label = 'Predicted data')
plt.title('Machine  prediction on discountionus function')
plt.legend()


fig = plt.figure()
plt.hist(dd, bins=100,color = 'k', label = 'difference betwen true and predicted data')
plt.title('difference betwen true and predicteddata')
plt.legend()


fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(ypredict,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Machine prediction  on discountionus function ')