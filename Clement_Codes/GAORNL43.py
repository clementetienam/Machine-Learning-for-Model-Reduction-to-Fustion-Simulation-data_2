# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:54:02 2019

@author: Dr Clement Etienam
Plot labels
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


import matplotlib as mpl

plt.title("Two informative features, one cluster per class", fontsize='small')

plt.scatter(X[:, 0], X[:, 1], marker='o', c=np.ravel(dd))
plt.legend()
plt.show()