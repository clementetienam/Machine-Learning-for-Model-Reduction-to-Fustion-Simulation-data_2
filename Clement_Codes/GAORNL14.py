# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:40:52 2019

@author: mjkiqce3
"""
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

X = open("inputactive.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(1000,1), 'F')  


y = open("outputactive.out") #533051 by 28
y = np.fromiter(y,float)
y=np.reshape(y,(1000,1), 'F') 

ytest = open("outputtestactive.out") #533051 by 28
ytest = np.fromiter(ytest,float)


xtest = open("inputtestactive.out") #533051 by 28
xtest = np.fromiter(xtest,float)
# k means determine k

matrix=np.concatenate((X,y), axis=1)
X=matrix
distortions = []
K = range(1,14)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()




kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
dd=kmeans.labels_

 
ff=matrix[dd==0]
ff2=matrix[dd==1]
ff3=matrix[dd==2]
ff4=matrix[dd==3]



xxx=np.reshape(matrix[0,:],(1,2))

centers=kmeans.cluster_centers_

ff=kmeans.predict(xxx)
#   predict for the determined points
#xHat = np.linspace(min(x), max(x), num=10000)
#yHat = myPWLF.predict(xHat)
