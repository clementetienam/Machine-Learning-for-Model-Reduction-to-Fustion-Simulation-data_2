# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 22 12:05:47 2019

@author: Dr Clement Etienam
This is the code for learning a machine for discountinous TGLF function
We will cluster th data first, use that label from the cluster and learn a
classifier then a regressor
This code is very important for TGLF-Passive learning
"""
from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier
import numpy as np
import scipy.io as sio
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
#------------------Active Learning Code----------------#
#trainset_size=5
#
#def run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,X_test,y_test):
#    from sklearn.metrics import accuracy_score
#    # build the model on training data
#    random_state = check_random_state(0)
#    initial_labeled_samples=5
#    permutation = np.random.choice(trainset_size,initial_labeled_samples,replace=False)
#    X_train = X_labeled[permutation]
#    y_train = y_labeled[permutation]
#    X_train = X_train.reshape((X_train.shape[0], -1))
#    queried = initial_labeled_samples
#    
#    X_val = np.array([])
#    y_val = np.array([])
#    X_val = np.copy(X_unlabeled)
#    X_val = np.delete(X_val, permutation, axis=0)
#    y_val = np.copy(y_oracle)
#    y_val = np.delete(y_val, permutation, axis=0)
#    
#    scaler = MinMaxScaler()
#    
#    X_train = scaler.fit_transform(X_train)
#    X_val   = scaler.transform(X_val)
#    X_test  = scaler.transform(X_test)
#    X_train = scaler.inverse_transform(X_train)
#    X_val   = scaler.inverse_transform(X_val)
#    X_test  = scaler.inverse_transform(X_test)
#    
#    model.fit(X_train, y_train )
#    
#    
#    
#    
#    max_queried=50000
#    ff_array = np.array([])
#    queried_array2 = np.array([])
#   
#    queried_array = np.empty((numrowstest, 0))
#    while queried < max_queried:
##            active_iteration += 1
#            probas_val = model.predict_proba(X_val)
#            rev = np.sort(probas_val, axis=1)[:, ::-1]
#            values = rev[:, 0] - rev[:, 1]
#            selection = np.argsort(values)[:initial_labeled_samples]
#            uncertain_samples=selection
#            
#            X_train = scaler.inverse_transform(X_train)
#            X_val   = scaler.inverse_transform(X_val)
#            X_test  = scaler.inverse_transform(X_test)
#            X_train = scaler.fit_transform(X_train)
#            X_val   = scaler.transform(X_val)
#            X_test  = scaler.transform(X_test)
#            
#            X_train = np.concatenate((X_train, X_val[uncertain_samples]))
#            y_train = np.concatenate((y_train, y_val[uncertain_samples]))
#            
#            X_val = np.delete(X_val, uncertain_samples, axis=0)
#            y_val = np.delete(y_val, uncertain_samples, axis=0)
#    
#            
#            X_train = scaler.inverse_transform(X_train)
#            X_val   = scaler.inverse_transform(X_val)
#            X_test  = scaler.inverse_transform(X_test)
#            X_train = scaler.fit_transform(X_train)
#            X_val   = scaler.transform(X_val)
#            X_test  = scaler.transform(X_test)
#            queried += initial_labeled_samples
#    
#    # make predictions for test data
#            model.fit(X_train, y_train)
#            
##            filename = 'finalizedclasschi_model.sav' #Save the classification model
##            pickle.dump(model, open(filename, 'wb')) #Save it
#            
#            labelDA = model.predict(X_test)
#            cm = confusion_matrix(y_test, labelDA,
#                          labels=model.classes_)
#            labelDAA=np.reshape(labelDA,(-1,1))
#            queried_array = np.append(queried_array, labelDAA, axis=1)
##            from sklearn.metrics import accuracy_score
#            ff=accuracy_score(y_test, labelDA)*100
#            ff_array = np.append(ff_array, ff)
#            queried_array2 = np.append(queried_array2, queried)
##            label_array = np.append(label_array, labelDA)
#            print('The accuracy is',ff)
#            print('Finished querying',queried,'points')
#            print("Confusion matrix after query",queried)
#            print(cm)
##            ff.append(ff)
#   
##        return selection
#
#    return labelDAA,ff_array,queried_array,queried_array2
def run_model(model,xxx,yyy,inputtest):
    # build the model on training data
    model.fit(xxx, yyy )

    # make predictions for test data
    labelDA = model.predict(inputtest)
#    ff=accuracy_score(yyytest, labelDA)*100
#    cm = confusion_matrix(yyytest, labelDA,
#                         labels=model.classes_)
#    print('The accuracy is',ff)
#    print("Confusion matrix after query")
#    print(cm)
    return labelDA
#------------------Begin Code----------------#
sgsim = open("orso.out") #533051 by 28
sgsim = np.fromiter(sgsim,float)

data = np.reshape(sgsim,(385747,28), 'F')
print('Standardize and normalize the input data')
input1=data[:,0:22]
output=data[:,22:29]

scaler = MinMaxScaler(feature_range=(0, 1))
#input1 = scaler.fit_transform(input1)
input1=np.arcsinh(input1)
output=np.arcsinh(output)
input11=input1
numrows=len(input1)    # rows of inout
numcols = len(input1[0])
inputtrain=(input1[0:300000,:]) #select the first 300000 for training
inputtest=(input1[300001:numrows,:]) #use the remaining data for testing

print('For the first output')
outputtrain=np.reshape((output[0:300000,0]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,0]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((inputtrain,outputtrain), axis=1)
ruuth = np.int(input("Enter the maximum number of clusters you want to accomodate: ") )

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
#dd=ddd[0:300000]
#ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
#model = RandomForestClassifier(n_estimators=100)
model = MLPClassifier(solver= 'lbfgs',max_iter=5000,validation_fraction=0.05)
print('Predict the classes from the classifier for test data')
labelDA=run_model(model,inputtrainclass,outputtrainclass,inputtest)
#-------------------Regression----------------#
from sklearn.neural_network import MLPRegressor
print('Learn regression of the 5 clusters with different labels from k-means ' )
print('set the output matrix')
clementanswer1=np.zeros((numrowstest,1))
for i in range(ruuth):
    label0=(np.asarray(np.where(dd == i))).T

    
##
    model0 = MLPRegressor(solver= 'lbfgs',max_iter=5000,validation_fraction=0.05)
   
    a0=inputtrainclass[label0,:]
    a0=np.reshape(a0,(-1,22),'F')

    b0=ydami[label0,:]
    b0=np.reshape(b0,(-1,1),'F')
    if a0.shape[0]!=0 and b0.shape[0]!=0:
       model0.fit(a0, b0)

    labelDA0=(np.asarray(np.where(labelDA == i))).T

##----------------------##------------------------##
    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,22),'F')
    if a00.shape[0]!=0:
       clementanswer1[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))

print('For the second output')
outputtrain=np.reshape((output[0:300000,1]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,1]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((inputtrain,outputtrain), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
#dd=ddd[0:300000]
#ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
#from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = MLPClassifier(solver= 'lbfgs',max_iter=5000)
print('Predict the classes from the classifier for test data')
labelDA=run_model(model,inputtrainclass,outputtrainclass,inputtest)
#-------------------Regression----------------#
print('Learn regression of the clusters with different labels from k-means ' )
clementanswer2=np.zeros((numrowstest,1))

for i in range(ruuth):
    label1=(np.asarray(np.where(dd == i))).T

   
##
    model1 = MLPRegressor(solver= 'lbfgs',max_iter=5000)
    a1=inputtrainclass[label1,:]
    a1=np.reshape(a1,(-1,22),'F')

    b1=ydami[label1,:]
    b1=np.reshape(b1,(-1,1),'F')
    if a1.shape[0]!=0 and b1.shape[0]!=0:
       model1.fit(a1, b1)

    labelDA1=(np.asarray(np.where(labelDA == i))).T

##----------------------##------------------------##
    a11=inputtest[labelDA1,:]
    a11=np.reshape(a11,(-1,22),'F')
    if a11.shape[0]!=0:
       clementanswer2[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

print('for the 3rd output')

outputtrain=np.reshape((output[0:300000,2]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,2]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((inputtrain,outputtrain), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
#dd=ddd[0:300000]
#ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
#from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
#model = RandomForestClassifier(n_estimators=100)
print('Predict the classes from the classifier for test data')
model = MLPClassifier(solver= 'lbfgs',max_iter=5000)
print('Predict the classes from the classifier for test data')
labelDA=run_model(model,inputtrainclass,outputtrainclass,inputtest)
#-------------------Regression----------------#
clementanswer3=np.zeros((numrowstest,1))
for i in range(ruuth):
    label2=(np.asarray(np.where(dd == i))).T

 
##
    model2 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
    a2=inputtrainclass[label2,:]
    a2=np.reshape(a2,(-1,22),'F')

    b2=ydami[label2,:]
    b2=np.reshape(b2,(-1,1),'F')
    if a2.shape[0]!=0 and b2.shape[0]!=0:
       model2.fit(a2, b2)

    labelDA2=(np.asarray(np.where(labelDA == i))).T

##----------------------##------------------------##
    a22=inputtest[labelDA2,:]
    a22=np.reshape(a22,(-1,22),'F')
    if a22.shape[0]!=0:
       clementanswer3[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

print('For the 4th output')
outputtrain=np.reshape((output[0:300000,3]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,3]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((inputtrain,outputtrain), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
#dd=ddd[0:300000]
#ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
#from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = MLPClassifier(solver= 'lbfgs',max_iter=5000)
print('Predict the classes from the classifier for test data')
labelDA=run_model(model,inputtrainclass,outputtrainclass,inputtest)
#-------------------Regression----------------#
clementanswer4=np.zeros((numrowstest,1))
for i in range(ruuth):
    label3=(np.asarray(np.where(dd == i))).T


##
    model3 = MLPRegressor(solver= 'lbfgs',max_iter=5000)
    a3=inputtrainclass[label3,:]
    a3=np.reshape(a3,(-1,22),'F')

    b3=ydami[label3,:]
    b3=np.reshape(b3,(-1,1),'F')
    if a3.shape[0]!=0 and b3.shape[0]!=0:
       model3.fit(a3, b3)

    labelDA3=(np.asarray(np.where(labelDA == i))).T

##----------------------##------------------------##
    a33=inputtest[labelDA3,:]
    a33=np.reshape(a33,(-1,22),'F')
    if a33.shape[0]!=0:
       clementanswer4[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))

print('For the 5th output')

outputtrain=np.reshape((output[0:300000,4]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,4]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((inputtrain,outputtrain), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
#dd=ddd[0:300000]
#ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
#ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
#from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = MLPClassifier(solver= 'lbfgs',max_iter=5000)
print('Predict the classes from the classifier for test data')
labelDA=run_model(model,inputtrainclass,outputtrainclass,inputtest)
#-------------------Regression----------------#
clementanswer5=np.zeros((numrowstest,1))
for i in range(ruuth):
    label4=(np.asarray(np.where(dd == i))).T


##
    model4 = MLPRegressor(solver= 'lbfgs',max_iter=5000)
    a4=inputtrainclass[label4,:]
    a4=np.reshape(a4,(-1,22),'F')

    b4=ydami[label4,:]
    b4=np.reshape(b4,(-1,1),'F')
    if a4.shape[0]!=0 and b4.shape[0]!=0:
       model4.fit(a4, b4)

    labelDA4=(np.asarray(np.where(labelDA == i))).T

##----------------------##------------------------##
    a44=inputtest[labelDA4,:]
    a44=np.reshape(a44,(-1,22),'F')
    if a44.shape[0]!=0:
       clementanswer5[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,1))

print('fot the 6th output')

outputtrain=np.reshape((output[0:300000,5]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,5]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((inputtrain,outputtrain), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T

dd=np.reshape(dd,(-1,1))

#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
#from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = MLPClassifier(solver= 'lbfgs',max_iter=5000)
print('Predict the classes from the classifier for test data')
labelDA=run_model(model,inputtrainclass,outputtrainclass,inputtest)
#-------------------Regression----------------#
clementanswer6=np.zeros((numrowstest,1))
for i in range(ruuth):
    label5=(np.asarray(np.where(dd == i))).T

##
    model5 = MLPRegressor(solver= 'lbfgs',max_iter=5000)
    a5=inputtrainclass[label5,:]
    a5=np.reshape(a5,(-1,22),'F')

    b5=ydami[label5,:]
    b5=np.reshape(b5,(-1,1),'F')
    if a5.shape[0]!=0 and b5.shape[0]!=0:
       model5.fit(a5, b5)

    labelDA5=(np.asarray(np.where(labelDA == i))).T

##----------------------##------------------------##
    a55=inputtest[labelDA5,:]
    a55=np.reshape(a55,(-1,22),'F')
    if a55.shape[0]!=0:
       clementanswer6[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,1))

#print(' Compute L2 and R2 for the machine')
#
#outputtest = np.reshape(outputtest, (numrowstest, 1))
#Lerrorsparse=(LA.norm(outputtest-clementanswer)/LA.norm(outputtest))**0.5
#L_2sparse=1-(Lerrorsparse**2)
##Coefficient of determination
#outputreq=np.zeros((numrowstest,1))
#for i in range(numrowstest):
#    outputreq[i,:]=outputtest[i,:]-np.mean(outputtest)
#
#
##outputreq=outputreq.T
#CoDspa=1-(LA.norm(outputtest-clementanswer)/LA.norm(outputreq))
#CoDsparse=1 - (1-CoDspa)**2 ;
#print ('R2 of fit using the machine is :', CoDsparse)
#print ('L2 of fit using the machine is :', L_2sparse)

outputteste=(output[300001:numrows,:])


print('Plot figures')
fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,1)
plt.plot(outputteste[0:50000,0], color = 'red', label = 'Real data')
plt.plot(clementanswer1[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_STRESS_TOR_i', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()



fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,2)
plt.plot(outputteste[0:50000,1], color = 'red', label = 'Real data')
plt.plot(clementanswer2[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_ENERGY_FLUX_i', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,3)
plt.plot(outputteste[0:50000,2], color = 'red', label = 'Real data')
plt.plot(clementanswer3[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_PARTICLE_FLUX_2', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,4)
plt.plot(outputteste[0:50000,3], color = 'red', label = 'Real data')
plt.plot(clementanswer4[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_PARTICLE_FLUX_1', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,5)
plt.plot(outputteste[0:50000,4], color = 'red', label = 'Real data')
plt.plot(clementanswer5[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_ENERGY_FLUX_1', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(10000),0,(10000)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,6)
plt.plot(outputteste[0:50000,5], color = 'red', label = 'Real data')
plt.plot(clementanswer6[0:50000,:], color = 'blue', label = 'Predicted data from Machine')

plt.title('OUT_tur_PARTICLE_FLUX_3', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
#plt.axis([0,(100),0,(100)])
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
plt.legend()
plt.show()


#
print('Plot figures')
fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,1)
plt.scatter(clementanswer1[0:50000,:],outputteste[0:50000,0], color ='c')
plt.title('output 1', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
#plt.axis([0,(100),0,(100)])
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,2)
plt.scatter(clementanswer2[0:50000,:],outputteste[0:50000,1], color ='c')
plt.title('output 2', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()



fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,3)
plt.scatter(clementanswer3[0:50000,:],outputteste[0:50000,2], color ='c')
plt.title('output 3', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()




fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,4)
plt.scatter(clementanswer4[0:50000,:],outputteste[0:50000,3], color ='c')
plt.title('output 4', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()

fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,5)
plt.scatter(clementanswer5[0:50000,:],outputteste[0:50000,4], color ='c')
plt.title('output 5', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,6)
plt.scatter(clementanswer6[0:50000,:],outputteste[0:50000,5], color ='c')
plt.title('output 6', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()

print('end of program')


