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

#!/usr/bin/env python
from mpi4py import MPI
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

plt.switch_backend('agg')

from numpy import linalg as LA

from sklearn.utils import check_random_state

from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from Forwarding2 import ensembleforwarding
from clementsurrogate import surrogateforward
from FASTTRANCCR import passiveclement
from sklearn.model_selection import train_test_split

print('Determine the optimum number of clusers')
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
trainset_size=5
filename = 'finalizedclasstauth_model.sav' #Save the classification model

def run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,X_test,y_test):
    from sklearn.metrics import accuracy_score
    # build the model on training data
    random_state = check_random_state(0)
    initial_labeled_samples=5
    permutation = np.random.choice(trainset_size,initial_labeled_samples,replace=False)
    X_train = X_labeled[permutation]
    y_train = y_labeled[permutation]
    X_train = X_train.reshape((X_train.shape[0], -1))
    queried = initial_labeled_samples
    
    X_val = np.array([])
    y_val = np.array([])
    
    X_valclement=np.copy(X_unlabeled)
    X_val = np.copy(X_unlabeled)
    
    X_val = np.delete(X_val, permutation, axis=0)
    X_valclement = np.delete(X_valclement, permutation, axis=0)
    
    y_val = np.copy(y_oracle)
    y_val = np.delete(y_val, permutation, axis=0)
    
    scaler = MinMaxScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    X_train = scaler.inverse_transform(X_train)
    X_val   = scaler.inverse_transform(X_val)
    X_test  = scaler.inverse_transform(X_test)
    
    model.fit(X_train, y_train )  
    
    max_queried=1000
    mark=np.int((max_queried/trainset_size)-1)
    ff_array = np.array([])
    queried_array2 = np.array([])
   
    queried_array = np.empty((numrows, 0))
    queried_clement = np.empty((trainset_size*numcols, 0))
    
    while queried < max_queried:
#            active_iteration += 1
            probas_val = model.predict_proba(X_val)
            rev = np.sort(probas_val, axis=1)[:, ::-1]
            values = rev[:, 0] - rev[:, 1]
            selection = np.argsort(values)[:initial_labeled_samples]
            uncertain_samples=selection
            
            X_train = scaler.inverse_transform(X_train)
            X_val   = scaler.inverse_transform(X_val)
            X_test  = scaler.inverse_transform(X_test)
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            X_test  = scaler.transform(X_test)
            
            X_train = np.concatenate((X_train, X_val[uncertain_samples]))
            y_train = np.concatenate((y_train, y_val[uncertain_samples]))
            
            X_val = np.delete(X_val, uncertain_samples, axis=0)
            y_val = np.delete(y_val, uncertain_samples, axis=0)
    
            
            X_train = scaler.inverse_transform(X_train)
            X_val   = scaler.inverse_transform(X_val)
            X_test  = scaler.inverse_transform(X_test)
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            X_test  = scaler.transform(X_test)
            queried += initial_labeled_samples
    
    # make predictions for test data
            model.fit(X_train, y_train)
#            pickle.dump(model, open(filename, 'wb')) #Save it
            labelDA = model.predict(X_test)
            cm = confusion_matrix(y_test, labelDA,
                          labels=model.classes_)
            labelDAA=np.reshape(labelDA,(-1,1))
            queried_array = np.append(queried_array, labelDAA, axis=1)
            uncertainmark=X_valclement[uncertain_samples]
            queried_clement = np.append(queried_clement, np.reshape(uncertainmark,(-1,1),'F'), axis=1)
#            from sklearn.metrics import accuracy_score
            ff=accuracy_score(y_test, labelDA)*100
            ff_array = np.append(ff_array, ff)
            queried_array2 = np.append(queried_array2, queried)
            #Uncertainclement are the points we dont know
#            label_array = np.append(label_array, labelDA)
            print('The accuracy is',ff)
            print('Finished querying',queried,'points')
            print("Confusion matrix after query",queried)
            print(cm)
#            ff.append(ff)
   
#        return selection

    return labelDAA,ff_array,queried_array,queried_array2,queried_clement,mark
    


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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
inputtrain=np.log(input[:2100,:]) #select the first 2100 for training
inputini=inputtrain
inputtest=np.log(input[2100:-1,:]) #use the remaining data for testing
output=test['tauth']
outputtrain=np.log(output[:2100,:]) #select the first 2100 for training
outputini=outputtrain
    
outputtest=(output[2100:-1,:]) #use the remaining data for testing
numrowstest=len(outputtest)
modelDNN = surrogateforward(inputini,outputini) #The DN surrogate model
for iclement in range (iterr):
    print('Starting Iteration %d'%iclement)
   
   
    numrows=len(inputtrain)    # rows of inout
    numcols = len(input[0]) # columns of input
#Start the loop to dynamically enrich the training set for sparse data coverage

  
    #model = MLPClassifier(solver= 'lbfgs',max_iter=5000)
    model=RandomForestClassifier(n_estimators=500)
    matrix=np.concatenate((inputtrain,outputtrain), axis=1)
    matrixtest=np.concatenate((inputtest,outputtest), axis=1)
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
    Xini=inputtrain
    
    print('Do for the test data as well')
    
#    kmeanstest =KMeans(n_clusters=nclusters,max_iter=100).fit(matrixtest)
#    ddtest=kmeanstest.labels_
#    ddtest=ddtest.T
#    ddtest=np.reshape(ddtest,(numrowstest,1))
    
    X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(inputtrain, dd, test_size=0.99)
    print('Enter the active learning dynamics')
    labelDAA,ff_array,queried_array,queried_array2,queried_clement,mark=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,inputtrain,dd)
    unie=np.amax(ff_array)
    ff_array=np.reshape(ff_array,(-1,1))

    label0=(np.asarray(np.where(ff_array==unie))).T
    finallabel=queried_array[:,label0[0,0]]

    labelDA=queried_array[:,-1]
    print('The highest accuracy is',unie,'at query point',label0[0,0])
#    loaded_model = pickle.load(open(filename, 'rb'))
#    result = loaded_model.score(inputtest, y_test)
#    print('The accuracy from reloaded saved model is',result)
    print('generate a distribution about this uncertain points') 
    #Routine to run a forward code of the uncertain points
    (xensemble) = ensembleforwarding(queried_clement,Ne)
    print('Save the data')
    np.savetxt('newdata.out', np.reshape(xensemble,(-1,1),'F'), fmt = '%4.6f', newline = '\n')
    #Augment this points
    #Pause here and use the newdata.out to get new points, (reshape the newdata
    # To be np.reshape(newdata(-1,9))) 
    xensemble=np.reshape(xensemble,(trainset_size,numcols,mark))
    for i in range(mark):
        inputtrain = np.concatenate((inputtrain, xensemble[:,:,i]))
    #Get yensemble from the forwarding and reshape doing 
    #yensemble=np.reshape(yensemble,(-1,1)'F')
    #y_keepit = np.concatenate((y_keepit, yensemble))
    np.savetxt('newtrainingset.out', np.reshape(inputtrain,(-1,1),'F'), fmt = '%4.6f', newline = '\n')
   
    print('For now use a DNN surrogate model for forwarding')
    
    outputtrain = modelDNN.predict(inputtrain)
    #outputtrain=np.concatenate((outputtrain, yensemble))
    print('Finished Iteration %d'%iclement)
    
print('now d some predictions')
(clementanswer) = passiveclement(inputtrain,outputtrain,inputtest,outputtest)
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

fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(2,2,1)
plt.scatter(clementanswer[0:2100],outputtest[0:2100], color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction on Fastran')
fig1.add_subplot(2,2,2)
plt.plot(outputtest[0:2100,:], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:2100,:], color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction on Fasttran')
plt.legend()
plt.show()

print('program executed')
