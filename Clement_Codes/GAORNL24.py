# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 19:20:42 2019

@author: mjkiqce3
Get the number of clusters for toy function
"""
from __future__ import print_function
print(__doc__)
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
X = open("inpiecewise.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(10000,2), 'F') 
 
y = open("outpiecewise.out") #533051 by 28
y = np.fromiter(y,float)
y = np.reshape(y,(10000,1), 'F')  


matrix=np.concatenate((X,y), axis=1)

kmeans = MiniBatchKMeans(n_clusters=20,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
#aa=kmeans.cluster_centers_
#
#yy=kmeans.predict(inputtest)
distortions = []
K = range(1,25)
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