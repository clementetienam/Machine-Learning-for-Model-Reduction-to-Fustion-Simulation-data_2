# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:16:06 2019

@author: Dr Clement Etienam

Gap statistics for determining optimum number of K for K-means
TGLF data
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
#------------------Begin Code----------------#

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
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=(output[300001:numrows,:])
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((inputtrain,outputtrain), axis=1)
def optimalK(data, nrefs, maxClusters):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


k, gapdf = optimalK(matrix, nrefs=5, maxClusters=25)
print ('Optimal k is: ', k)

plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()
