# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 08:37:42 2019
Getting the right number of clusters for TGLF data
@author: Dr Clement Etienam
This code is important for TGLF
"""
from __future__ import print_function
print(__doc__)
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

sgsim = open("Finaldata.out") #533051 by 28
sgsim = np.fromiter(sgsim,float)

data = np.reshape(sgsim,(533051,28), 'F')
print('Standardize and normalize the input data')
input1=data[:,0:22]
output=data[:,22:29]

scaler = MinMaxScaler(feature_range=(0, 1))
input1 = scaler.fit_transform(input1)

input11=input1
numrows=len(input1)    # rows of inout
numcols = len(input1[0])
inputtrain=(input1[0:300000,:]) #select the first 300000 for training
inputtest=(input1[300001:numrows,:]) #use the remaining data for testing
outputtrain=(output[0:300000,:]) #select the first 300000 for training
#outputtrain=np.arcsinh(outputtrain)
outputtest=(output[300001:numrows,:])
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((inputtrain,outputtrain), axis=1)

kmeans = MiniBatchKMeans(n_clusters=5,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
#aa=kmeans.cluster_centers_
#
#yy=kmeans.predict(inputtest)
distortions = []
K = range(1,30)
for k in K:
    kmeanModel = MiniBatchKMeans(n_clusters=k,random_state=0,batch_size=6,max_iter=10).fit(matrix)

    kmeanModel.fit(matrix)
    distortions.append(sum(np.min(cdist(matrix, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / matrix.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
