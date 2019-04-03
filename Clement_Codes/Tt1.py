# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:35:53 2019

@author: Dr Clement Etienam
"""


import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential


X = open("inputtianke.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(500,6), 'F')  


y = open("outputtianke.out") #533051 by 28
y = np.fromiter(y,float)
y=np.reshape(y,(500,1), 'F') 

ytest = open("outputtianketest.out") #533051 by 28
ytest = np.fromiter(ytest,float)
ytest=np.reshape(ytest,(500,1), 'F') 

xtest = open("inputtianketest.out") #533051 by 28
xtest = np.fromiter(xtest,float)
xtest = np.reshape(xtest,(500,6), 'F')  
# k means determine k

print('With artificial neural network')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(solver= 'lbfgs',max_iter=100)
a0=X


b0=y
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

yes=model0.predict(xtest)


print('Use Deep neural network with Keras')

np.random.seed(7)
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(200, activation = 'relu', input_dim = 6))

# Adding the second hidden layer
model.add(Dense(units = 420, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 21, activation = 'relu'))

# Adding the output layer

model.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X, y, batch_size = 10, epochs = 1000)

yann = model.predict(xtest)


print('do for ridge regression')
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit(X,y) 
yrig=reg.predict(xtest)

print('Do Bayesian ridge regression')
regb = linear_model.BayesianRidge()
regb.fit(X, y)
ybrig=regb.predict(xtest)



print('Do polynomial regression')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

modelpp = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])


modelpp = modelpp.fit(X, y)
ypo=modelpp.predict(xtest)

from sklearn.linear_model import OrthogonalMatchingPursuit

regomp = OrthogonalMatchingPursuit().fit(X, y)

yomp=reg.predict(xtest)

print('Do Linear Regression')
from sklearn import linear_model
regl = linear_model.LinearRegression()
regl.fit(X,y)                                     
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                 normalize=False)
ylinear=regl.predict(xtest)

print('Plot figures')

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(yes,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction of oil price in $ (Multi-layer Perceptron)')

fig = plt.figure()
plt.plot(ytest, color = 'red', label = 'Real data')
plt.plot(yes, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction of oil price in $ (Muli-Layer Perceptron)')
plt.ylabel('oil price in $',fontsize = 13)
plt.xlabel('Time(days)',fontsize = 13)
plt.legend()
plt.show()

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(yann,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction of oil price in $(Deep Neural Network-Keras)')

fig = plt.figure()
plt.plot(ytest, color = 'red', label = 'Real data')
plt.plot(yann, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction of oil price in $ (Deep Neural Network-Keras)')
plt.ylabel('oil price in $',fontsize = 13)
plt.xlabel('Time(days)',fontsize = 13)
plt.legend()
plt.show()


fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(yrig,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction of oil price in $(ridge rehression)')

fig = plt.figure()
plt.plot(ytest, color = 'red', label = 'Real data')
plt.plot(yrig, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction of oil price in $ (ridge regression)')
plt.ylabel('oil price in $',fontsize = 13)
plt.xlabel('Time(days)',fontsize = 13)
plt.legend()
plt.show()

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(ybrig,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction of oil price in $(Bayesian ridge rehression)')

fig = plt.figure()
plt.plot(ytest, color = 'red', label = 'Real data')
plt.plot(ybrig, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction of oil price in $ (Bayesian ridge regression)')
plt.ylabel('oil price in $',fontsize = 13)
plt.xlabel('Time(days)',fontsize = 13)
plt.legend()
plt.show()

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(ypo,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction of oil price in $(polynomial regression)')

fig = plt.figure()
plt.plot(ytest, color = 'red', label = 'Real data')
plt.plot(ypo, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction of oil price in $ (polynomial regression)')
plt.ylabel('oil price in $',fontsize = 13)
plt.xlabel('Time(days)',fontsize = 13)
plt.legend()
plt.show()

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(yomp,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction of oil price in $(orthogonal matching pursuit)')

fig = plt.figure()
plt.plot(ytest, color = 'red', label = 'Real data')
plt.plot(yomp, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction of oil price in $ (orthogonal matching pursuit)')
plt.ylabel('oil price in $',fontsize = 13)
plt.xlabel('Time(days)',fontsize = 13)
plt.legend()
plt.show()

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(ylinear,ytest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction of oil price in $(Linear Regression)')

fig = plt.figure()
plt.plot(ytest, color = 'red', label = 'Real data')
plt.plot(ylinear, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction of oil price in $ (Linear Regression)')
plt.ylabel('oil price in $',fontsize = 13)
plt.xlabel('Time(days)',fontsize = 13)
plt.legend()
plt.show()