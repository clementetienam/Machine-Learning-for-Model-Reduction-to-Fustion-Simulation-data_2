# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 05 12:05:47 2019

@author: Dr Clement Etienam
This is the code for learning a machine for discountinous Toy function
We will cluster th data first, use that label from the cluster and learn a
classifier then a regressor
This code is very important
"""
from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linalg as LA

#------------------Begin Code----------------#
start_time = datetime.now()
x = np.genfromtxt("chi_itg.dat",skip_header=1, dtype='float')

#for i in range (numrows):
#    if y[i]==0:
#        y2[i]=-1
##    
#    elif y[i]>0:
#        y2[i]=1
print('cluster with X and y')
X = open("inputactive.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(1000,1), 'F') 
 
y = open("outputactive.out") #533051 by 28
y = np.fromiter(y,float)
y = np.reshape(y,(1000,1), 'F')  

ydami=y

ytest = open("outputtestactive.out") #533051 by 28
ytest = np.fromiter(ytest,float)
ytest = np.reshape(ytest,(1000,1), 'F')  

outputtest=ytest

numrowstest=len(outputtest)

inputtrainclass=X
#
outputtrainclass=y
matrix=np.concatenate((X,y), axis=1)

xtest = open("inputtestactive.out") #533051 by 28
xtest = np.fromiter(xtest,float)
xtest = np.reshape(xtest,(1000,1), 'F')  

inputtest=xtest
print('Do the K-means clustering with 8 clusters of [X,y] and get the labels')
kmeans =KMeans(n_clusters=4,max_iter=100).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(1000,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=X
outputtrainclass=np.reshape(dd,(1000,1))


#%outputtest=y(290000+1:end,:);
def run_model(model):
    # build the model on training data
    model.fit(inputtrainclass, outputtrainclass )

    # make predictions for test data
    labelDA = model.predict(inputtest)
    return labelDA
print(' Learn the classifer from the predicted labels from Kmeans')
model = MLPClassifier(solver= 'lbfgs',max_iter=2000)
print('Predict the classes from the classifier for test data')
labelDA=run_model(model)
#-------------------Regression----------------#
print('Learn regression of the 5 clusters with different labels from k-means ' )
label0=(np.asarray(np.where(dd == 0))).T
label1=(np.asarray(np.where(dd == 1))).T
label2=(np.asarray(np.where(dd == 2))).T
label3=(np.asarray(np.where(dd == 3))).T


print('set the output matrix')
clementanswer=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(solver= 'lbfgs',max_iter=2000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,1),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(solver= 'lbfgs',max_iter=2000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,1),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,1),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(solver= 'lbfgs',max_iter=2000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,1),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,1),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(solver= 'lbfgs',max_iter=2000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,1),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,1),'F')
model3.fit(a3, b3)



print('Time for the prediction')
labelDA0=(np.asarray(np.where(labelDA == 0))).T
labelDA1=(np.asarray(np.where(labelDA == 1))).T
labelDA2=(np.asarray(np.where(labelDA == 2))).T
labelDA3=(np.asarray(np.where(labelDA == 3))).T

##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,1),'F')
clementanswer[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,1),'F')
clementanswer[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,1),'F')
clementanswer[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,1),'F')
clementanswer[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))



print(' Compute L2 and R2 for the machine')

outputtest = np.reshape(outputtest, (numrowstest, 1))
Lerrorsparse=(LA.norm(outputtest-clementanswer)/LA.norm(outputtest))**0.5
L_2sparse=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest[i,:]-np.mean(outputtest)


#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest-clementanswer)/LA.norm(outputreq))
CoDsparse=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine is :', CoDsparse)
print ('L2 of fit using the machine is :', L_2sparse)

print('Plot figures')

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(clementanswer,outputtest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')


fig = plt.figure()
plt.plot(outputtest, color = 'red', label = 'Real data')
plt.plot(clementanswer, color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction on Chi')
plt.legend()
plt.show()



