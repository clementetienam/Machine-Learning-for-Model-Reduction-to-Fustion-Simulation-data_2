# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 00:06:12 2019

@author: mjkiqce3
"""

import numpy as np
import pwlf


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
y = np.reshape(y,(10000), 'F')  

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

#
#xtest = open("intestpiecewise.out") #533051 by 28
#xtest = np.fromiter(xtest,float)
#xtest = np.reshape(xtest,(10000,1), 'F')  
#
xtest=X
inputtest=xtest

Xuse=np.reshape(X[:,0],(10000))

x0 = np.array([ -10, -8,-6,0,5,8,9])
myPWLF = pwlf.PiecewiseLinFit(Xuse,y)
myPWLF.fit_with_breaks(x0)

#   fit the data for four line segments
res = myPWLF.fit(12)

#   predict for the determined points

yHat = myPWLF.predict(Xuse)
import matplotlib.pyplot as plt
fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(yHat ,y, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')


fig = plt.figure()
plt.plot(y, color = 'red', label = 'Real data')
plt.plot(yHat, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction on toy function')
plt.legend()
plt.show()














#
#y = np.array([  0.00000000e+00,   9.69801700e-03,   2.94350340e-02,
#         4.39052750e-02,   5.45343950e-02,   6.74104940e-02,
#         8.34831790e-02,   1.02580042e-01,   1.22767939e-01,
#         1.42172312e-01,   0.00000000e+00,   8.58600000e-06,
#         8.31543400e-03,   2.34184100e-02,   3.39709150e-02,
#         4.03581990e-02,   4.53545600e-02,   5.02345260e-02,
#         5.55253360e-02,   6.14750770e-02,   6.82125120e-02,
#         7.55892510e-02,   8.38356810e-02,   9.26413070e-02,
#         1.02039790e-01,   1.11688258e-01,   1.21390666e-01,
#         1.31196948e-01,   0.00000000e+00,   1.56706510e-02,
#         3.54628780e-02,   4.63739040e-02,   5.61442590e-02,
#         6.78542550e-02,   8.16388310e-02,   9.77756110e-02,
#         1.16531753e-01,   1.37038283e-01,   0.00000000e+00,
#         1.16951050e-02,   3.12089850e-02,   4.41776550e-02,
#         5.42877590e-02,   6.63321350e-02,   8.07655920e-02,
#         9.70363280e-02,   1.15706975e-01,   1.36687642e-01,
#         0.00000000e+00,   1.50144640e-02,   3.44519970e-02,
#         4.55907760e-02,   5.59556700e-02,   6.88450940e-02,
#         8.41374060e-02,   1.01254006e-01,   1.20605073e-01,
#         1.41881288e-01,   1.62618058e-01])
#x = np.array([  0.00000000e+00,   8.82678000e-03,   3.25615100e-02,
#         5.66106800e-02,   7.95549800e-02,   1.00936330e-01,
#         1.20351520e-01,   1.37442010e-01,   1.51858250e-01,
#         1.64433570e-01,   0.00000000e+00,  -2.12600000e-05,
#         7.03872000e-03,   1.85494500e-02,   3.00926700e-02,
#         4.17617000e-02,   5.37279600e-02,   6.54941000e-02,
#         7.68092100e-02,   8.76596300e-02,   9.80525800e-02,
#         1.07961810e-01,   1.17305210e-01,   1.26063930e-01,
#         1.34180360e-01,   1.41725010e-01,   1.48629710e-01,
#         1.55374770e-01,   0.00000000e+00,   1.65610200e-02,
#         3.91016100e-02,   6.18679400e-02,   8.30997400e-02,
#         1.02132890e-01,   1.19011260e-01,   1.34620080e-01,
#         1.49429370e-01,   1.63539960e-01,  -0.00000000e+00,
#         1.01980300e-02,   3.28642800e-02,   5.59461900e-02,
#         7.81388400e-02,   9.84458400e-02,   1.16270210e-01,
#         1.31279040e-01,   1.45437090e-01,   1.59627540e-01,
#         0.00000000e+00,   1.63404300e-02,   4.00086000e-02,
#         6.34390200e-02,   8.51085900e-02,   1.04787860e-01,
#         1.22120350e-01,   1.36931660e-01,   1.50958760e-01,
#         1.65299640e-01,   1.79942720e-01])
