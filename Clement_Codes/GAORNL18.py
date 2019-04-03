# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 08:37:42 2019

@author: mjkiqce3
This code was used to determine number of clusters for chi-data
"""
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
x = np.genfromtxt("chi_itg.dat",skip_header=1, dtype='float')
df=x
test=df
numrows=len(test)    # rows of inout
X=np.log(df[:,0:10])
y=test[:,-1]
outputtest=y[290000:600001]
inputtrainclass=X[0:290000,:];
#
outputtrainclass=np.reshape(y[0:290000],(290000,1));
matrix=np.concatenate((inputtrainclass,outputtrainclass), axis=1)
inputtest=X[290000:numrows,:];
kmeans = MiniBatchKMeans(n_clusters=3,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
#aa=kmeans.cluster_centers_
#
#yy=kmeans.predict(inputtest)
distortions = []
K = range(1,14)
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