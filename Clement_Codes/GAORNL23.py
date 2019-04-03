# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 05 12:05:47 2019

@author: Dr Clement Etienam
This is the code for learning a machine for discountinous TGLF function
We will cluster th data first, use that label from the cluster and learn a
classifier then a regressor
This code is very important for TGLF
"""
from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA

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

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=10,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(300000,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(300000,1))

#%outputtest=y(290000+1:end,:);
def run_model(model):
    # build the model on training data
    model.fit(inputtrainclass, outputtrainclass )

    # make predictions for test data
    labelDA = model.predict(inputtest)
    return labelDA
print(' Learn the classifer from the predicted labels from Kmeans')
model = MLPClassifier(hidden_layer_sizes=(200, ),max_iter=1000)
#model = RandomForestClassifier(n_estimators=500)
print('Predict the classes from the classifier for test data')
labelDA=run_model(model)
#-------------------Regression----------------#
print('Learn regression of the 5 clusters with different labels from k-means ' )
label0=(np.asarray(np.where(dd == 0))).T
label1=(np.asarray(np.where(dd == 1))).T
label2=(np.asarray(np.where(dd == 2))).T
label3=(np.asarray(np.where(dd == 3))).T
label4=(np.asarray(np.where(dd == 4))).T
label5=(np.asarray(np.where(dd == 5))).T
label6=(np.asarray(np.where(dd == 6))).T
label7=(np.asarray(np.where(dd == 7))).T
label8=(np.asarray(np.where(dd == 8))).T
label9=(np.asarray(np.where(dd == 9))).T

print('set the output matrix')
clementanswer=np.zeros((numrowstest,6))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,22),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,6),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,22),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,6),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,22),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,6),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,22),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,6),'F')
model3.fit(a3, b3)

##
model4 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a4=inputtrainclass[label4,:]
a4=np.reshape(a4,(-1,22),'F')

b4=ydami[label4,:]
b4=np.reshape(b4,(-1,6),'F')
model4.fit(a4, b4)

##
model5 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a5=inputtrainclass[label5,:]
a5=np.reshape(a5,(-1,22),'F')

b5=ydami[label5,:]
b5=np.reshape(b5,(-1,6),'F')
model5.fit(a5, b5)

##
model6 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a6=inputtrainclass[label6,:]
a6=np.reshape(a6,(-1,22),'F')

b6=ydami[label6,:]
b6=np.reshape(b6,(-1,6),'F')
model6.fit(a6, b6)

##
model7 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a7=inputtrainclass[label7,:]
a7=np.reshape(a7,(-1,22),'F')

b7=ydami[label7,:]
b7=np.reshape(b7,(-1,6),'F')
model7.fit(a7, b7)

##
model8 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a8=inputtrainclass[label8,:]
a8=np.reshape(a8,(-1,22),'F')

b8=ydami[label8,:]
b8=np.reshape(b8,(-1,6),'F')
model8.fit(a8, b8)

##
model9 = MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
a9=inputtrainclass[label9,:]
a9=np.reshape(a9,(-1,22),'F')

b9=ydami[label9,:]
b9=np.reshape(b9,(-1,6),'F')
model9.fit(a9, b9)


print('Time for the prediction')
labelDA0=(np.asarray(np.where(labelDA == 0))).T
labelDA1=(np.asarray(np.where(labelDA == 1))).T
labelDA2=(np.asarray(np.where(labelDA == 2))).T
labelDA3=(np.asarray(np.where(labelDA == 3))).T
labelDA4=(np.asarray(np.where(labelDA == 4))).T
labelDA5=(np.asarray(np.where(labelDA == 5))).T
labelDA6=(np.asarray(np.where(labelDA == 6))).T
labelDA7=(np.asarray(np.where(labelDA == 7))).T
labelDA8=(np.asarray(np.where(labelDA == 8))).T
labelDA9=(np.asarray(np.where(labelDA == 9))).T
##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,22),'F')
clementanswer[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,6))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,22),'F')
clementanswer[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,6))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,22),'F')
clementanswer[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,6))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,22),'F')
clementanswer[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,6))

a44=inputtest[labelDA4,:]
a44=np.reshape(a44,(-1,22),'F')
clementanswer[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,6))

a55=inputtest[labelDA5,:]
a55=np.reshape(a55,(-1,22),'F')
clementanswer[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,6))

a66=inputtest[labelDA6,:]
a66=np.reshape(a66,(-1,22),'F')
clementanswer[np.ravel(labelDA6),:]=np.reshape(model6.predict(a66),(-1,6))

a77=inputtest[labelDA7,:]
a77=np.reshape(a77,(-1,22),'F')
clementanswer[np.ravel(labelDA7),:]=np.reshape(model7.predict(a77),(-1,6))

a88=inputtest[labelDA8,:]
a88=np.reshape(a88,(-1,22),'F')
clementanswer[np.ravel(labelDA8),:]=np.reshape(model8.predict(a88),(-1,6))

a99=inputtest[labelDA9,:]
a99=np.reshape(a99,(-1,22),'F')
clementanswer[np.ravel(labelDA9),:]=np.reshape(model9.predict(a99),(-1,6))
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

print('Plot figures')
fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,1)
plt.plot(outputtest[0:500,0], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:500,0], color = 'blue', label = 'Predicted data from Machine')

plt.title('output 1', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.legend()
plt.show()



fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,2)
plt.plot(outputtest[0:500,1], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:500,1], color = 'blue', label = 'Predicted data from Machine')

plt.title('output 2', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,3)
plt.plot(outputtest[0:500,2], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:500,2], color = 'blue', label = 'Predicted data from Machine')

plt.title('output 3', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,4)
plt.plot(outputtest[0:500,3], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:500,3], color = 'blue', label = 'Predicted data from Machine')

plt.title('output 4', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,5)
plt.plot(outputtest[0:500,4], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:500,4], color = 'blue', label = 'Predicted data from Machine')

plt.title('output 5', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.legend()
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,6)
plt.plot(outputtest[0:500,5], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:500,5], color = 'blue', label = 'Predicted data from Machine')

plt.title('output 6', fontsize = 15)
plt.ylabel('Values',fontsize = 13)
plt.xlabel('Sample points',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.legend()
plt.show()


#
print('Plot figures')
fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,1)
plt.scatter(clementanswer[0:500,0],outputtest[0:500,0], color ='c')
plt.title('output 1', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])

plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,2)
plt.scatter(clementanswer[0:500,1],outputtest[0:500,1], color ='c')
plt.title('output 2', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()



fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,3)
plt.scatter(clementanswer[0:500,2],outputtest[0:500,2], color ='c')
plt.title('output 3', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()




fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,4)
plt.scatter(clementanswer[0:500,3],outputtest[0:500,3], color ='c')
plt.title('output 4', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()

fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,5)
plt.scatter(clementanswer[0:500,4],outputtest[0:500,4], color ='c')
plt.title('output 5', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,6)
plt.scatter(clementanswer[0:500,5],outputtest[0:500,5], color ='c')
plt.title('output 6', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()



