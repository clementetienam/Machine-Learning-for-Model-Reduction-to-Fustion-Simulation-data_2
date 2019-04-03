# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:33:15 2019

@author: mjkiqce3
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


#for i in range (numrows):
#    if y[i]==0:
#        y2[i]=-1
##    
#    elif y[i]>0:
#        y2[i]=1
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
print('Do the K-means clustering with 18 clusters of [X,y] and get the labels')
kmeans =KMeans(n_clusters=18,max_iter=100).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(10000,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
y=dd

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_train = scaler.fit_transform(X_train)

X_test  = scaler.transform(X_test)
X_train = scaler.inverse_transform(X_train)
  
X_test  = scaler.inverse_transform(X_test)


#%outputtest=y(290000+1:end,:);
def run_model(model):
    # build the model on training data
    model.fit(X_train, y_train )

    # make predictions for test data
    labelDA = model.predict(X_test)
    return labelDA
print(' Learn the classifer from the predicted labels from Kmeans')
model = MLPClassifier(solver= 'lbfgs',max_iter=3000)
print('Predict the classes from the classifier for test data')

labelDA=run_model(model)
from sklearn.metrics import accuracy_score
ffout=accuracy_score(y_test, labelDA)*100
print('The accuracy is',ffout)