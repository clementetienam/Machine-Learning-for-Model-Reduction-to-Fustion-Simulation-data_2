# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:50:59 2019

@author: mjkiqce3
"""
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

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


model = Sequential()
model.add(Dense(1300, input_dim=2, activation='relu'))

model.add(Dense(700, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(units = 1))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=2000, batch_size=10,  verbose=2)
scores = model.evaluate(X, y)
ynn = model.predict(X)


print(' Compute L2 and R2 for the machine')
numrowstest=len(y)
from numpy import linalg as LA
outputtest = np.reshape(y, (numrowstest, 1))
Lerrorsparse=(LA.norm(outputtest-ynn)/LA.norm(outputtest))**0.5
L_2sparse=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest[i,:]-np.mean(outputtest)


#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest-ynn)/LA.norm(outputreq))
CoDsparse=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine is :', CoDsparse)
print ('L2 of fit using the machine is :', L_2sparse)

print('Plot figures')

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(ynn,outputtest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')


fig = plt.figure()
plt.plot(outputtest, color = 'red', label = 'Real data')
plt.plot(ynn, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction on toy function')
plt.legend()
plt.show()