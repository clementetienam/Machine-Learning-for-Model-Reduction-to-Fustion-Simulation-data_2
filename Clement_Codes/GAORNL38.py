# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 00:27:02 2019

@author: mjkiqce3
"""


print(__doc__)
import numpy as np
from piecewise.regressor import piecewise
import matplotlib.pyplot as plt

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


Xuse=np.reshape(X[:,0],(10000,1))

model = piecewise(Xuse, y)
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
