# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:34:45 2019

@author: Dr Clement Etienam
Supevisor :Professor Kody Law
Collaborator: Dr Mark Cianciosa
Train CCR with an Active learning aproach on the Tauth data
-------------------------------------------------------------------------------
This is the Logic
1)We wil first detrmine the optimum number of clusters suitable for the data
2)Do K-means clustering of the input-output pair
3)Train a classifier with the input and the labels from step (2)
4)Sample those points with low probability
5) Generate more points around these uncertain points
6) Get the coresponding simulated (y) data of these generated points
7) Add these points and retrain the machine
-------------------------------------------------------------------------------
"""
#Import the necessary libraries
from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from numpy import linalg as LA

from sklearn.utils import check_random_state

from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from Forwarding2 import ensembleforwarding
print('Determine he opimum number of clusers')
def optimalK(data, nrefs, maxClusters):
    """
    Calculates KMeans optimal K using Gap Statistic 
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

print('Load the Tauth/Fastran data')
max_clusters = np.int(input("Enter the maximum number of clusters you want to accomodate: ") )
iterr = np.int(input("Enter the maximum number of iteration you want to run: ") )
Ne= np.int(input("Enter the number of ensemble you want for pertubation: ") )
import scipy.io as sio
test = sio.loadmat('JM_tauth_data')
r0=test['r0']
a0=test['a0']
kappa=test['kappa']
delta=test['delta']
ip=test['ip']
b0=test['b0']
nebar=test['nebar']
zeff=test['zeff']
ploss=test['ploss']
#input=np.concatenate((r0, a0,kappa,delta,ip,b0,nebar,zeff,ploss), axis=1)
input=np.concatenate((test['r0'], test['a0'],test['kappa'],test['delta'],test['ip'],test['b0'],test['nebar'],test['zeff'],test['ploss']), axis=1)
# second input is more efficient
numrows=len(input)    # rows of inout
numcols = len(input[0]) # columns of input
inputtrain=np.log(input[:2100,:]) #select the first 2100 for training
inputtest=np.log(input[2100:numrows,:]) #use the remaining data for testing
output=test['tauth']
outputtrain=np.log(output[:2100,:]) #select the first 2100 for training
outputtest=(output[2100:numrows,:]) #use the remaining data for testing
numrowstest=len(outputtest)


from sklearn.preprocessing import MinMaxScaler
#Start the loop to dynamically enrich the training set for sparse data coverage
for iclement in range(iterr):
    print('starting Iteration %d'%iclement)
    numrows=len(inputtrain) 
    model = RandomForestClassifier(n_estimators=1000)
    matrix=np.concatenate((inputtrain,outputtrain), axis=1)
    # use Gap statistics to get optimum number of clustsrs
    k, gapdf = optimalK(matrix, nrefs=5, maxClusters=max_clusters)
    nclusters=k
    print ('Optimal k is: ', nclusters)
    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    plt.show()
    #Now do the K-means clustering
    print('Do the K-means clustering of [X,y] and get the labels')
    kmeans =KMeans(n_clusters=nclusters,max_iter=100).fit(matrix)
    dd=kmeans.labels_
    dd=dd.T
    dd=np.reshape(dd,(numrows,1))
    scaler = MinMaxScaler()
#-------------------#---------------------------------#
    print('Use the labels to train a classifier')
    inputtrainclass=inputtrain
    X_keepit = np.array([])
    X_keepit = np.copy(inputtrain)
    y_keepit = np.array([])
    y_keepit = np.copy(outputtrain)
    outputtrainclass=np.reshape(dd,(numrows,1))
    inputtrain = scaler.fit_transform(inputtrain)
    inputtrain = scaler.inverse_transform(inputtrain) 
    #Fit the clasifier
    model.fit(inputtrain, dd )
    #Locate those points with low probability in the set
    probas_val = model.predict_proba(inputtrain)
    rev = np.sort(probas_val, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    initial_labeled_samples=numrows
    selection = np.argsort(values)[:initial_labeled_samples]
    uncertain_samples=selection
            
    inputtrain = scaler.inverse_transform(inputtrain)
  
    inputtrain = scaler.fit_transform(inputtrain)
  
    #Although redundant information view the augmented set        
    inputtrain = np.concatenate((inputtrain, inputtrain[uncertain_samples]))
    dd = np.concatenate((dd, dd[uncertain_samples]))
    X_keepituncertain=X_keepit[uncertain_samples] 
    print('generate a distribution about this uncertain points') 
    #Routine to run a forward code of the uncertain points
    (xensemble) = ensembleforwarding(X_keepituncertain,Ne)
    np.savetxt('newdata.out', np.reshape(xensemble,(-1,1),'F'), fmt = '%4.6f', newline = '\n')
    #Augment this points
    #Pause here and use the newdata.out to get new points, (reshape the newdata
    # To be np.reshape(newdata(-1,9)))     
    X_keepit = np.concatenate((X_keepit, xensemble))
    #Get yensemble from the forwarding and reshape doing 
    #yensemble=np.reshape(yensemble,(-1,1)'F')
    #y_keepit = np.concatenate((y_keepit, yensemble))
    
    inputtrain=X_keepit
    outputtrain=y_keepit
    print('Finished Iteration %d'%iclement)
