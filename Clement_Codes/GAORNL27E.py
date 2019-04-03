# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:15:20 2019

@author: Dr Clement Etienam
Active learning very important
"""

from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

trainset_size=5

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
    
    
    
    
    max_queried=3000
    ff_array = np.array([])
    queried_array2 = np.array([])
   
    queried_array = np.empty((5000, 0))
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
            
            labelDA = model.predict(X_test)
            cm = confusion_matrix(y_test, labelDA,
                          labels=model.classes_)
            labelDA=np.reshape(labelDA,(-1,1))
            queried_array = np.append(queried_array, labelDA, axis=1)
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

    return labelDA,ff_array,queried_array,queried_array2
    
print(' Learn the classifer from the predicted labels from Kmeans')
model = RandomForestClassifier(n_estimators=500)
#model = MLPClassifier(solver= 'lbfgs',max_iter=3000)
#model=LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=30)
print('Predict the classes from the classifier for test data')
print('cluster with X and y')



X = open("inpiecewise.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(10000,2), 'F') 

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
nclusters = np.int(input("Enter the number of clusters you want: ") )
print('Do the K-means clustering with 18 clusters of [X,y] and get the labels')
kmeans =KMeans(n_clusters=nclusters,max_iter=100).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(10000,1))
y=dd
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=X
outputtrainclass=np.reshape(dd,(10000,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(X_train, y_train, test_size=0.999)

labelDA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,X_test,y_test)

unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]
print('The highest accuracy is',unie,'at query point',label0[0,0])


from sklearn.metrics import accuracy_score
ffout=accuracy_score(y_test, labelDA)*100



XX, YY = np.meshgrid(np.arange(100),np.arange(100))

fig1 = plt.figure(figsize =(8,8))
machine1=np.reshape(labelDA,(100,100),'F')
JM1=np.reshape(y_test,(100,100),'F')


fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
ax.plot(queried_array2,ff_array)
ax.set_ylim(bottom=0, top=100)
ax.yaxis.grid(True, linestyle='--', alpha=1/2)
ax.set_title('Incremental classification accuracy in %')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy in %')
plt.show()


fig1.add_subplot(2,2,1)
plt.pcolormesh(XX.T,YY.T,machine1,cmap = 'jet')
plt.title('Machine', fontsize = 15)
plt.ylabel('X2',fontsize = 13)
plt.xlabel('X1',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar15 = plt.colorbar()
cbar15.ax.set_ylabel('2D function machine',fontsize = 13)
plt.clim(-1,1)

fig1.add_subplot(2,2,2)
plt.pcolormesh(XX.T,YY.T,JM1,cmap = 'jet')
plt.title('True', fontsize = 15)
plt.ylabel('X2',fontsize = 13)
plt.xlabel('X1',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar15 = plt.colorbar()
cbar15.ax.set_ylabel('2D function true',fontsize = 13)
plt.clim(-1,1)