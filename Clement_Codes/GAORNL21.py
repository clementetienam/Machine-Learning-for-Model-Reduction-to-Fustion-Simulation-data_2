# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 05 12:05:47 2019

@author: Dr Clement Etienam
This is the code for learning a machine for discountinous Chi function
We will cluster th data first, use that label from the cluster and learn a
classifier then a regressor
This code is very important
"""
from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linalg as LA

#------------------Begin Code----------------#
start_time = datetime.now()
x = np.genfromtxt("chi_itg.dat",skip_header=1, dtype='float')

#for i in range (numrows):
#    if y[i]==0:
#        y2[i]=-1
##    
#    elif y[i]>0:
#        y2[i]=1
print('cluster with X and y')
df=x
test=df
numrows=len(test)    # rows of inout
X=np.log(df[:,0:10])
y=test[:,-1]
ydami=np.reshape(y[0:290000],(290000,1));
outputtest=y[290000:600001]
numrowstest=len(outputtest)
inputtrainclass=X[0:290000,:]
#
outputtrainclass=np.reshape(y[0:290000],(290000,1))
matrix=np.concatenate((inputtrainclass,outputtrainclass), axis=1)
inputtest=X[290000:numrows,:]
print('Do the K-means clustering with 8 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=8,random_state=0,batch_size=6,max_iter=10).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(290000,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=X[0:290000,:]
outputtrainclass=np.reshape(dd,(290000,1))

inputtest=X[290000:numrows,:];
#%outputtest=y(290000+1:end,:);
def run_model(model):
    # build the model on training data
    model.fit(inputtrainclass, outputtrainclass )

    # make predictions for test data
    labelDA = model.predict(inputtest)
    return labelDA
print(' Learn the classifer from the predicted labels from Kmeans')
model = MLPClassifier(max_iter=1000)
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

print('set the output matrix')
clementanswer=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(max_iter=1000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,10),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(max_iter=1000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,10),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,1),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(max_iter=1000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,10),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,1),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(max_iter=1000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,10),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,1),'F')
model3.fit(a3, b3)

##
model4 = MLPRegressor(max_iter=1000)
a4=inputtrainclass[label4,:]
a4=np.reshape(a4,(-1,10),'F')

b4=ydami[label4,:]
b4=np.reshape(b4,(-1,1),'F')
model4.fit(a4, b4)

##
model5 = MLPRegressor(max_iter=1000)
a5=inputtrainclass[label5,:]
a5=np.reshape(a5,(-1,10),'F')

b5=ydami[label5,:]
b5=np.reshape(b5,(-1,1),'F')
model5.fit(a5, b5)

##
model6 = MLPRegressor(max_iter=1000)
a6=inputtrainclass[label6,:]
a6=np.reshape(a6,(-1,10),'F')

b6=ydami[label6,:]
b6=np.reshape(b6,(-1,1),'F')
model6.fit(a6, b6)

##
model7 = MLPRegressor(max_iter=1000)
a7=inputtrainclass[label7,:]
a7=np.reshape(a7,(-1,10),'F')

b7=ydami[label7,:]
b7=np.reshape(b7,(-1,1),'F')
model7.fit(a7, b7)

print('Time for the prediction')
labelDA0=(np.asarray(np.where(labelDA == 0))).T
labelDA1=(np.asarray(np.where(labelDA == 1))).T
labelDA2=(np.asarray(np.where(labelDA == 2))).T
labelDA3=(np.asarray(np.where(labelDA == 3))).T
labelDA4=(np.asarray(np.where(labelDA == 4))).T
labelDA5=(np.asarray(np.where(labelDA == 5))).T
labelDA6=(np.asarray(np.where(labelDA == 6))).T
labelDA7=(np.asarray(np.where(labelDA == 7))).T
##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,10),'F')
clementanswer[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,10),'F')
clementanswer[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,10),'F')
clementanswer[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,10),'F')
clementanswer[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))

a44=inputtest[labelDA4,:]
a44=np.reshape(a44,(-1,10),'F')
clementanswer[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,1))

a55=inputtest[labelDA5,:]
a55=np.reshape(a55,(-1,10),'F')
clementanswer[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,1))

a66=inputtest[labelDA6,:]
a66=np.reshape(a66,(-1,10),'F')
clementanswer[np.ravel(labelDA6),:]=np.reshape(model6.predict(a66),(-1,1))

a77=inputtest[labelDA7,:]
a77=np.reshape(a77,(-1,10),'F')
clementanswer[np.ravel(labelDA7),:]=np.reshape(model7.predict(a77),(-1,1))

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
plt.plot(outputtest[0:500,:], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:500,:], color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction on Chi')
plt.legend()
plt.show()


