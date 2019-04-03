# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 05 12:05:47 2019

@author: Dr Clement Etienam
CCR-Active learning
"""

from __future__ import print_function
print(__doc__)

#from IPython import get_ipython
#get_ipython().magic('reset -sf')

#from sklearn.neural_network import MLPClassifier
import numpy as np

#from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
#from sklearn import metrics
#from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#from datetime import datetime
from numpy import linalg as LA
#from sklearn.svm import SVC
from sklearn.utils import check_random_state
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix
#------------------Begin Code----------------#
trainset_size=5
filename = 'finalizedclass2Dtoy_model.sav' #Save the classification model

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
    X_val = np.copy(X_unlabeled)
    X_val = np.delete(X_val, permutation, axis=0)
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
    
    
    
    
    max_queried=200
    ff_array = np.array([])
    queried_array2 = np.array([])
   
    queried_array = np.empty((1000, 0))
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
            pickle.dump(model, open(filename, 'wb')) #Save it
            labelDA = model.predict(X_test)
            cm = confusion_matrix(y_test, labelDA,
                          labels=model.classes_)
            labelDAA=np.reshape(labelDA,(-1,1))
            queried_array = np.append(queried_array, labelDAA, axis=1)
#            from sklearn.metrics import accuracy_score
            ff=accuracy_score(y_test, labelDA)*100
            ff_array = np.append(ff_array, ff)
            queried_array2 = np.append(queried_array2, queried)
#            label_array = np.append(label_array, labelDA)
            print('The accuracy is',ff)
            print('Finished querying',queried,'points')
            print("Confusion matrix after query",queried)
            print(cm)
#            ff.append(ff)
   
#        return selection

    return labelDAA,ff_array,queried_array,queried_array2
    
print(' Learn the classifer from the predicted labels from Kmeans')
model = RandomForestClassifier(n_estimators=1000)

print('cluster with X and y')
X = open("inputtestactive.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(1000,1), 'F') 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
(scaler.fit(X))
X=(scaler.transform(X))
y = open("outputtestactive.out") #533051 by 28
y = np.fromiter(y,float)
y = np.reshape(y,(1000,1), 'F')  

ydami=y

print('split for regression prblem') 
from sklearn.model_selection import train_test_split
#X_traind, inputtest, y_traind, outputtest = train_test_split(X, y, test_size=0.5)
#ytest=y
outputtest=y
X_traind=X
y_traind=y
inputtest=X
numrowstest=len(outputtest)

#inputtrainclass=X
##
#outputtrainclass=y
matrix=np.concatenate((X_traind,y_traind), axis=1)


#
#xtest = open("intestpiecewise.out") #533051 by 28
#xtest = np.fromiter(xtest,float)
#xtest = np.reshape(xtest,(10000,1), 'F')  
#
#xtest=X
#inputtest=xtest
nclusters = np.int(input("Enter the number of clusters you want: ") )
print('Do the K-means clustering with 18 clusters of [X,y] and get the labels')
kmeans =KMeans(n_clusters=nclusters,max_iter=100).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(1000,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=X_traind
outputtrainclass=np.reshape(dd,(1000,1))


print('Split for classifier problem')
#X_train, X_test, y_train, y_test = train_test_split(X, dd, test_size=0.5)

X_train=X_traind
X_test=inputtest
y_train=dd
y_test=dd

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(X_train, y_train, test_size=0.8)

labelDAA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,X_test,y_test)
unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]

labelDA=queried_array[:,-1]
print('The highest accuracy is',unie,'at query point',label0[0,0])
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(inputtest, y_test)
print('The accuracy from reloaded saved model is',result)
#-------------------Regression----------------#
print('Learn regression of the clusters with different labels from k-means ' )
print('set the output matrix')
clementanswer=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
for i in range(nclusters):
    label0=(np.asarray(np.where(y_train == i))).T


##
    model0 = MLPRegressor(solver= 'lbfgs',max_iter=5000)
    a0=X_train[label0,:]
    a0=np.reshape(a0,(-1,1),'F')

    b0=y_traind[label0,:]
    b0=np.reshape(b0,(-1,1),'F')
    model0.fit(a0, b0)




    print('Time for the prediction')
    labelDA0=(np.asarray(np.where(labelDA == i))).T


##----------------------##------------------------##
    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,1),'F')
    clementanswer[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))

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
plt.title('Prediction on toy function')
plt.legend()
plt.show()


#XX, YY = np.meshgrid(np.arange(100),np.arange(100))
#
#fig1 = plt.figure(figsize =(8,8))
#machine1=np.reshape(clementanswer,(100,100),'F')
#JM1=np.reshape(outputtest,(100,100),'F')
#
#
#
#fig1.add_subplot(2,2,1)
#plt.pcolormesh(XX.T,YY.T,machine1,cmap = 'jet')
#plt.title('Machine', fontsize = 15)
#plt.ylabel('2',fontsize = 11)
#plt.xlabel('1',fontsize = 11)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
#cbar15 = plt.colorbar()
#cbar15.ax.set_ylabel('2D function',fontsize = 11)
#plt.clim(0,20)
#
#fig1.add_subplot(2,2,2)
#plt.pcolormesh(XX.T,YY.T,JM1,cmap = 'jet')
#plt.title('True', fontsize = 15)
#plt.ylabel('2',fontsize = 11)
#plt.xlabel('1',fontsize = 11)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
#cbar15 = plt.colorbar()
#cbar15.ax.set_ylabel('2D function',fontsize = 11)
#plt.clim(0,20)
#
#
#
