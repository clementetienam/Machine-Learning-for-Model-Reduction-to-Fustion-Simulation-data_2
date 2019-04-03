# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 21 12:05:47 2019

@author: Dr Clement Etienam
This is the code for learning a machine for discountinous TGLF function
We will cluster th data first, use that label from the cluster and learn a
classifier then a regressor
This code is very important for TGLF-Active learning
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
#------------------Active Learning Code----------------#
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
    
    
    
    
    max_queried=50000
    ff_array = np.array([])
    queried_array2 = np.array([])
   
    queried_array = np.empty((numrowstest, 0))
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
            
#            filename = 'finalizedclasschi_model.sav' #Save the classification model
#            pickle.dump(model, open(filename, 'wb')) #Save it
            
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

#------------------Begin Code----------------#
sgsim = open("TGLGmum.out") #533051 by 28
sgsim = np.fromiter(sgsim,float)

data = np.reshape(sgsim,(310991,28), 'F')
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

print('For the first output')
outputtrain=np.reshape((output[0:300000,0]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,0]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((input11,np.reshape(output[:,0],(-1,1))), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=24,random_state=0,batch_size=6,max_iter=10).fit(matrix)
ddd=kmeans.labels_
ddd=ddd.T
dd=ddd[0:300000]
ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
#model = RandomForestClassifier(n_estimators=100)
model = MLPClassifier(solver= 'lbfgs',max_iter=3000)
print('Predict the classes from the classifier for test data')
from sklearn.model_selection import train_test_split

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(inputtrainclass, outputtrainclass, test_size=0.999)

labelDAA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,inputtest,ddtest)
unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]

labelDA=queried_array[:,-1]
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
label10=(np.asarray(np.where(dd == 10))).T
label11=(np.asarray(np.where(dd == 11))).T
label12=(np.asarray(np.where(dd == 12))).T
label13=(np.asarray(np.where(dd == 13))).T
label14=(np.asarray(np.where(dd == 14))).T
label15=(np.asarray(np.where(dd == 15))).T
label16=(np.asarray(np.where(dd == 16))).T
label17=(np.asarray(np.where(dd == 17))).T
label18=(np.asarray(np.where(dd == 18))).T
label19=(np.asarray(np.where(dd == 19))).T
label20=(np.asarray(np.where(dd == 20))).T
label21=(np.asarray(np.where(dd == 21))).T
label22=(np.asarray(np.where(dd == 22))).T
label23=(np.asarray(np.where(dd == 23))).T



print('set the output matrix')
clementanswer1=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,22),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,22),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,1),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,22),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,1),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,22),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,1),'F')
model3.fit(a3, b3)

##
model4 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a4=inputtrainclass[label4,:]
a4=np.reshape(a4,(-1,22),'F')

b4=ydami[label4,:]
b4=np.reshape(b4,(-1,1),'F')
model4.fit(a4, b4)

##
model5 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a5=inputtrainclass[label5,:]
a5=np.reshape(a5,(-1,22),'F')

b5=ydami[label5,:]
b5=np.reshape(b5,(-1,1),'F')
model5.fit(a5, b5)

##
model6 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a6=inputtrainclass[label6,:]
a6=np.reshape(a6,(-1,22),'F')

b6=ydami[label6,:]
b6=np.reshape(b6,(-1,1),'F')
model6.fit(a6, b6)

##
model7 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a7=inputtrainclass[label7,:]
a7=np.reshape(a7,(-1,22),'F')

b7=ydami[label7,:]
b7=np.reshape(b7,(-1,1),'F')
model7.fit(a7, b7)

##
model8 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a8=inputtrainclass[label8,:]
a8=np.reshape(a8,(-1,22),'F')

b8=ydami[label8,:]
b8=np.reshape(b8,(-1,1),'F')
model8.fit(a8, b8)

##
model9 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a9=inputtrainclass[label9,:]
a9=np.reshape(a9,(-1,22),'F')

b9=ydami[label9,:]
b9=np.reshape(b9,(-1,1),'F')
model9.fit(a9, b9)

##
model10 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a10=inputtrainclass[label10,:]
a10=np.reshape(a10,(-1,22),'F')

b10=ydami[label10,:]
b10=np.reshape(b10,(-1,1),'F')
model10.fit(a10, b10)

##
model11 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a11=inputtrainclass[label11,:]
a11=np.reshape(a11,(-1,22),'F')

b11=ydami[label11,:]
b11=np.reshape(b11,(-1,1),'F')
model11.fit(a11, b11)

##
model12 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a12=inputtrainclass[label12,:]
a12=np.reshape(a11,(-1,22),'F')

b12=ydami[label12,:]
b12=np.reshape(b12,(-1,1),'F')
model12.fit(a12, b12)


##
model13 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a13=inputtrainclass[label13,:]
a13=np.reshape(a13,(-1,22),'F')

b13=ydami[label13,:]
b13=np.reshape(b13,(-1,1),'F')
model13.fit(a13, b13)

##
model14 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a14=inputtrainclass[label14,:]
a14=np.reshape(a14,(-1,22),'F')

b14=ydami[label14,:]
b14=np.reshape(b14,(-1,1),'F')
model14.fit(a14, b14)

##
model15 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a15=inputtrainclass[label15,:]
a15=np.reshape(a15,(-1,22),'F')

b15=ydami[label15,:]
b15=np.reshape(b15,(-1,1),'F')
model15.fit(a15, b15)

##
model16 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a16=inputtrainclass[label16,:]
a16=np.reshape(a16,(-1,22),'F')

b16=ydami[label16,:]
b16=np.reshape(b16,(-1,1),'F')
model16.fit(a16, b16)

##
model17 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a17=inputtrainclass[label17,:]
a17=np.reshape(a17,(-1,22),'F')

b17=ydami[label17,:]
b17=np.reshape(b17,(-1,1),'F')
model17.fit(a17, b17)

##
model18 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a18=inputtrainclass[label18,:]
a18=np.reshape(a18,(-1,22),'F')

b18=ydami[label18,:]
b18=np.reshape(b18,(-1,1),'F')
model18.fit(a18, b18)

##
model19 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a19=inputtrainclass[label19,:]
a19=np.reshape(a19,(-1,22),'F')

b19=ydami[label19,:]
b19=np.reshape(b19,(-1,1),'F')
model19.fit(a19, b19)

##
model20 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a20=inputtrainclass[label20,:]
a20=np.reshape(a20,(-1,22),'F')

b20=ydami[label20,:]
b20=np.reshape(b20,(-1,1),'F')
model20.fit(a20, b20)

##
model21 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a21=inputtrainclass[label21,:]
a21=np.reshape(a21,(-1,22),'F')

b21=ydami[label21,:]
b21=np.reshape(b21,(-1,1),'F')
model21.fit(a21, b21)

##
model22 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a22=inputtrainclass[label22,:]
a22=np.reshape(a22,(-1,22),'F')

b22=ydami[label22,:]
b22=np.reshape(b22,(-1,1),'F')
model22.fit(a22, b22)

##
model23 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a23=inputtrainclass[label23,:]
a23=np.reshape(a23,(-1,22),'F')

b23=ydami[label23,:]
b23=np.reshape(b23,(-1,1),'F')
model23.fit(a23, b23)

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
labelDA10=(np.asarray(np.where(labelDA == 10))).T
labelDA11=(np.asarray(np.where(labelDA == 11))).T
labelDA12=(np.asarray(np.where(labelDA == 12))).T
labelDA13=(np.asarray(np.where(labelDA == 13))).T
labelDA14=(np.asarray(np.where(labelDA == 14))).T
labelDA15=(np.asarray(np.where(labelDA == 15))).T
labelDA16=(np.asarray(np.where(labelDA == 16))).T
labelDA17=(np.asarray(np.where(labelDA == 17))).T
labelDA18=(np.asarray(np.where(labelDA == 18))).T
labelDA19=(np.asarray(np.where(labelDA == 19))).T
labelDA20=(np.asarray(np.where(labelDA == 20))).T
labelDA21=(np.asarray(np.where(labelDA == 21))).T
labelDA22=(np.asarray(np.where(labelDA == 22))).T
labelDA23=(np.asarray(np.where(labelDA == 23))).T

##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,22),'F')
clementanswer1[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,22),'F')
clementanswer1[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,22),'F')
clementanswer1[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,22),'F')
clementanswer1[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))

a44=inputtest[labelDA4,:]
a44=np.reshape(a44,(-1,22),'F')
clementanswer1[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,1))

a55=inputtest[labelDA5,:]
a55=np.reshape(a55,(-1,22),'F')
clementanswer1[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,1))

a66=inputtest[labelDA6,:]
a66=np.reshape(a66,(-1,22),'F')
clementanswer1[np.ravel(labelDA6),:]=np.reshape(model6.predict(a66),(-1,1))

a77=inputtest[labelDA7,:]
a77=np.reshape(a77,(-1,22),'F')
clementanswer1[np.ravel(labelDA7),:]=np.reshape(model7.predict(a77),(-1,1))

a88=inputtest[labelDA8,:]
a88=np.reshape(a88,(-1,22),'F')
clementanswer1[np.ravel(labelDA8),:]=np.reshape(model8.predict(a88),(-1,1))

a99=inputtest[labelDA9,:]
a99=np.reshape(a99,(-1,22),'F')
clementanswer1[np.ravel(labelDA9),:]=np.reshape(model9.predict(a99),(-1,1))

a1010=inputtest[labelDA10,:]
a1010=np.reshape(a1010,(-1,22),'F')
clementanswer1[np.ravel(labelDA10),:]=np.reshape(model10.predict(a1010),(-1,1))

a1111=inputtest[labelDA11,:]
a1111=np.reshape(a1111,(-1,22),'F')
clementanswer1[np.ravel(labelDA11),:]=np.reshape(model11.predict(a1111),(-1,1))

a1212=inputtest[labelDA12,:]
a1212=np.reshape(a1212,(-1,22),'F')
clementanswer1[np.ravel(labelDA12),:]=np.reshape(model12.predict(a1212),(-1,1))

a1313=inputtest[labelDA13,:]
a1313=np.reshape(a1313,(-1,22),'F')
clementanswer1[np.ravel(labelDA13),:]=np.reshape(model13.predict(a1313),(-1,1))

a1414=inputtest[labelDA14,:]
a1414=np.reshape(a1414,(-1,22),'F')
clementanswer1[np.ravel(labelDA14),:]=np.reshape(model14.predict(a1414),(-1,1))

a1515=inputtest[labelDA15,:]
a1515=np.reshape(a1515,(-1,22),'F')
clementanswer1[np.ravel(labelDA15),:]=np.reshape(model15.predict(a1515),(-1,1))

a1616=inputtest[labelDA16,:]
a1616=np.reshape(a1616,(-1,22),'F')
clementanswer1[np.ravel(labelDA16),:]=np.reshape(model16.predict(a1616),(-1,1))

a1717=inputtest[labelDA17,:]
a1717=np.reshape(a1717,(-1,22),'F')
clementanswer1[np.ravel(labelDA17),:]=np.reshape(model17.predict(a1717),(-1,1))

a1818=inputtest[labelDA18,:]
a1818=np.reshape(a1818,(-1,22),'F')
clementanswer1[np.ravel(labelDA18),:]=np.reshape(model18.predict(a1818),(-1,1))

a1919=inputtest[labelDA19,:]
a1919=np.reshape(a1919,(-1,22),'F')
clementanswer1[np.ravel(labelDA19),:]=np.reshape(model19.predict(a1919),(-1,1))

a2020=inputtest[labelDA20,:]
a2020=np.reshape(a2020,(-1,22),'F')
clementanswer1[np.ravel(labelDA20),:]=np.reshape(model20.predict(a2020),(-1,1))

a2121=inputtest[labelDA21,:]
a2121=np.reshape(a2121,(-1,22),'F')
clementanswer1[np.ravel(labelDA21),:]=np.reshape(model21.predict(a2121),(-1,1))

a2222=inputtest[labelDA22,:]
a2222=np.reshape(a2222,(-1,22),'F')
clementanswer1[np.ravel(labelDA22),:]=np.reshape(model22.predict(a2222),(-1,1))

a2323=inputtest[labelDA23,:]
a2323=np.reshape(a2323,(-1,22),'F')
clementanswer1[np.ravel(labelDA23),:]=np.reshape(model23.predict(a2323),(-1,1))


print('For the second output')
outputtrain=np.reshape((output[0:300000,1]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,1]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((input11,np.reshape(output[:,1],(-1,1))), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=24,random_state=0,batch_size=6,max_iter=10).fit(matrix)
ddd=kmeans.labels_
ddd=ddd.T
dd=ddd[0:300000]
ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = RandomForestClassifier(n_estimators=100)
print('Predict the classes from the classifier for test data')
from sklearn.model_selection import train_test_split

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(inputtrainclass, outputtrainclass, test_size=0.999)

labelDAA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,inputtest,ddtest)
unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]

labelDA=queried_array[:,-1]
#-------------------Regression----------------#
print('Learn regression of the clusters with different labels from k-means ' )
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
label10=(np.asarray(np.where(dd == 10))).T
label11=(np.asarray(np.where(dd == 11))).T
label12=(np.asarray(np.where(dd == 12))).T
label13=(np.asarray(np.where(dd == 13))).T
label14=(np.asarray(np.where(dd == 14))).T
label15=(np.asarray(np.where(dd == 15))).T
label16=(np.asarray(np.where(dd == 16))).T
label17=(np.asarray(np.where(dd == 17))).T
label18=(np.asarray(np.where(dd == 18))).T
label19=(np.asarray(np.where(dd == 19))).T
label20=(np.asarray(np.where(dd == 20))).T
label21=(np.asarray(np.where(dd == 21))).T
label22=(np.asarray(np.where(dd == 22))).T
label23=(np.asarray(np.where(dd == 23))).T



print('set the output matrix')
clementanswer2=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,22),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,22),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,1),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,22),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,1),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,22),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,1),'F')
model3.fit(a3, b3)

##
model4 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a4=inputtrainclass[label4,:]
a4=np.reshape(a4,(-1,22),'F')

b4=ydami[label4,:]
b4=np.reshape(b4,(-1,1),'F')
model4.fit(a4, b4)

##
model5 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a5=inputtrainclass[label5,:]
a5=np.reshape(a5,(-1,22),'F')

b5=ydami[label5,:]
b5=np.reshape(b5,(-1,1),'F')
model5.fit(a5, b5)

##
model6 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a6=inputtrainclass[label6,:]
a6=np.reshape(a6,(-1,22),'F')

b6=ydami[label6,:]
b6=np.reshape(b6,(-1,1),'F')
model6.fit(a6, b6)

##
model7 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a7=inputtrainclass[label7,:]
a7=np.reshape(a7,(-1,22),'F')

b7=ydami[label7,:]
b7=np.reshape(b7,(-1,1),'F')
model7.fit(a7, b7)

##
model8 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a8=inputtrainclass[label8,:]
a8=np.reshape(a8,(-1,22),'F')

b8=ydami[label8,:]
b8=np.reshape(b8,(-1,1),'F')
model8.fit(a8, b8)

##
model9 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a9=inputtrainclass[label9,:]
a9=np.reshape(a9,(-1,22),'F')

b9=ydami[label9,:]
b9=np.reshape(b9,(-1,1),'F')
model9.fit(a9, b9)

##
model10 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a10=inputtrainclass[label10,:]
a10=np.reshape(a10,(-1,22),'F')

b10=ydami[label10,:]
b10=np.reshape(b10,(-1,1),'F')
model10.fit(a10, b10)

##
model11 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a11=inputtrainclass[label11,:]
a11=np.reshape(a11,(-1,22),'F')

b11=ydami[label11,:]
b11=np.reshape(b11,(-1,1),'F')
model11.fit(a11, b11)

##
model12 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a12=inputtrainclass[label12,:]
a12=np.reshape(a11,(-1,22),'F')

b12=ydami[label12,:]
b12=np.reshape(b12,(-1,1),'F')
model12.fit(a12, b12)


##
model13 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a13=inputtrainclass[label13,:]
a13=np.reshape(a13,(-1,22),'F')

b13=ydami[label13,:]
b13=np.reshape(b13,(-1,1),'F')
model13.fit(a13, b13)

##
model14 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a14=inputtrainclass[label14,:]
a14=np.reshape(a14,(-1,22),'F')

b14=ydami[label14,:]
b14=np.reshape(b14,(-1,1),'F')
model14.fit(a14, b14)

##
model15 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a15=inputtrainclass[label15,:]
a15=np.reshape(a15,(-1,22),'F')

b15=ydami[label15,:]
b15=np.reshape(b15,(-1,1),'F')
model15.fit(a15, b15)

##
model16 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a16=inputtrainclass[label16,:]
a16=np.reshape(a16,(-1,22),'F')

b16=ydami[label16,:]
b16=np.reshape(b16,(-1,1),'F')
model16.fit(a16, b16)

##
model17 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a17=inputtrainclass[label17,:]
a17=np.reshape(a17,(-1,22),'F')

b17=ydami[label17,:]
b17=np.reshape(b17,(-1,1),'F')
model17.fit(a17, b17)

##
model18 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a18=inputtrainclass[label18,:]
a18=np.reshape(a18,(-1,22),'F')

b18=ydami[label18,:]
b18=np.reshape(b18,(-1,1),'F')
model18.fit(a18, b18)

##
model19 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a19=inputtrainclass[label19,:]
a19=np.reshape(a19,(-1,22),'F')

b19=ydami[label19,:]
b19=np.reshape(b19,(-1,1),'F')
model19.fit(a19, b19)

##
model20 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a20=inputtrainclass[label20,:]
a20=np.reshape(a20,(-1,22),'F')

b20=ydami[label20,:]
b20=np.reshape(b20,(-1,1),'F')
model20.fit(a20, b20)

##
model21 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a21=inputtrainclass[label21,:]
a21=np.reshape(a21,(-1,22),'F')

b21=ydami[label21,:]
b21=np.reshape(b21,(-1,1),'F')
model21.fit(a21, b21)

##
model22 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a22=inputtrainclass[label22,:]
a22=np.reshape(a22,(-1,22),'F')

b22=ydami[label22,:]
b22=np.reshape(b22,(-1,1),'F')
model22.fit(a22, b22)

##
model23 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a23=inputtrainclass[label23,:]
a23=np.reshape(a23,(-1,22),'F')

b23=ydami[label23,:]
b23=np.reshape(b23,(-1,1),'F')
model23.fit(a23, b23)

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
labelDA10=(np.asarray(np.where(labelDA == 10))).T
labelDA11=(np.asarray(np.where(labelDA == 11))).T
labelDA12=(np.asarray(np.where(labelDA == 12))).T
labelDA13=(np.asarray(np.where(labelDA == 13))).T
labelDA14=(np.asarray(np.where(labelDA == 14))).T
labelDA15=(np.asarray(np.where(labelDA == 15))).T
labelDA16=(np.asarray(np.where(labelDA == 16))).T
labelDA17=(np.asarray(np.where(labelDA == 17))).T
labelDA18=(np.asarray(np.where(labelDA == 18))).T
labelDA19=(np.asarray(np.where(labelDA == 19))).T
labelDA20=(np.asarray(np.where(labelDA == 20))).T
labelDA21=(np.asarray(np.where(labelDA == 21))).T
labelDA22=(np.asarray(np.where(labelDA == 22))).T
labelDA23=(np.asarray(np.where(labelDA == 23))).T

##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,22),'F')
clementanswer2[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,22),'F')
clementanswer2[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,22),'F')
clementanswer2[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,22),'F')
clementanswer2[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))

a44=inputtest[labelDA4,:]
a44=np.reshape(a44,(-1,22),'F')
clementanswer2[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,1))

a55=inputtest[labelDA5,:]
a55=np.reshape(a55,(-1,22),'F')
clementanswer2[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,1))

a66=inputtest[labelDA6,:]
a66=np.reshape(a66,(-1,22),'F')
clementanswer2[np.ravel(labelDA6),:]=np.reshape(model6.predict(a66),(-1,1))

a77=inputtest[labelDA7,:]
a77=np.reshape(a77,(-1,22),'F')
clementanswer2[np.ravel(labelDA7),:]=np.reshape(model7.predict(a77),(-1,1))

a88=inputtest[labelDA8,:]
a88=np.reshape(a88,(-1,22),'F')
clementanswer2[np.ravel(labelDA8),:]=np.reshape(model8.predict(a88),(-1,1))

a99=inputtest[labelDA9,:]
a99=np.reshape(a99,(-1,22),'F')
clementanswer2[np.ravel(labelDA9),:]=np.reshape(model9.predict(a99),(-1,1))

a1010=inputtest[labelDA10,:]
a1010=np.reshape(a1010,(-1,22),'F')
clementanswer2[np.ravel(labelDA10),:]=np.reshape(model10.predict(a1010),(-1,1))

a1111=inputtest[labelDA11,:]
a1111=np.reshape(a1111,(-1,22),'F')
clementanswer2[np.ravel(labelDA11),:]=np.reshape(model11.predict(a1111),(-1,1))

a1212=inputtest[labelDA12,:]
a1212=np.reshape(a1212,(-1,22),'F')
clementanswer2[np.ravel(labelDA12),:]=np.reshape(model12.predict(a1212),(-1,1))

a1313=inputtest[labelDA13,:]
a1313=np.reshape(a1313,(-1,22),'F')
clementanswer2[np.ravel(labelDA13),:]=np.reshape(model13.predict(a1313),(-1,1))

a1414=inputtest[labelDA14,:]
a1414=np.reshape(a1414,(-1,22),'F')
clementanswer2[np.ravel(labelDA14),:]=np.reshape(model14.predict(a1414),(-1,1))

a1515=inputtest[labelDA15,:]
a1515=np.reshape(a1515,(-1,22),'F')
clementanswer2[np.ravel(labelDA15),:]=np.reshape(model15.predict(a1515),(-1,1))

a1616=inputtest[labelDA16,:]
a1616=np.reshape(a1616,(-1,22),'F')
clementanswer2[np.ravel(labelDA16),:]=np.reshape(model16.predict(a1616),(-1,1))

a1717=inputtest[labelDA17,:]
a1717=np.reshape(a1717,(-1,22),'F')
clementanswer2[np.ravel(labelDA17),:]=np.reshape(model17.predict(a1717),(-1,1))

a1818=inputtest[labelDA18,:]
a1818=np.reshape(a1818,(-1,22),'F')
clementanswer2[np.ravel(labelDA18),:]=np.reshape(model18.predict(a1818),(-1,1))

a1919=inputtest[labelDA19,:]
a1919=np.reshape(a1919,(-1,22),'F')
clementanswer2[np.ravel(labelDA19),:]=np.reshape(model19.predict(a1919),(-1,1))

a2020=inputtest[labelDA20,:]
a2020=np.reshape(a2020,(-1,22),'F')
clementanswer2[np.ravel(labelDA20),:]=np.reshape(model20.predict(a2020),(-1,1))

a2121=inputtest[labelDA21,:]
a2121=np.reshape(a2121,(-1,22),'F')
clementanswer2[np.ravel(labelDA21),:]=np.reshape(model21.predict(a2121),(-1,1))

a2222=inputtest[labelDA22,:]
a2222=np.reshape(a2222,(-1,22),'F')
clementanswer2[np.ravel(labelDA22),:]=np.reshape(model22.predict(a2222),(-1,1))

a2323=inputtest[labelDA23,:]
a2323=np.reshape(a2323,(-1,22),'F')
clementanswer2[np.ravel(labelDA23),:]=np.reshape(model23.predict(a2323),(-1,1))

print('for the 3rd output')

outputtrain=np.reshape((output[0:300000,2]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,2]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((input11,np.reshape(output[:,2],(-1,1))), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=24,random_state=0,batch_size=6,max_iter=10).fit(matrix)
ddd=kmeans.labels_
ddd=ddd.T
dd=ddd[0:300000]
ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = RandomForestClassifier(n_estimators=100)
print('Predict the classes from the classifier for test data')
from sklearn.model_selection import train_test_split

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(inputtrainclass, outputtrainclass, test_size=0.999)

labelDAA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,inputtest,ddtest)
unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]

labelDA=queried_array[:,-1]
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
label10=(np.asarray(np.where(dd == 10))).T
label11=(np.asarray(np.where(dd == 11))).T
label12=(np.asarray(np.where(dd == 12))).T
label13=(np.asarray(np.where(dd == 13))).T
label14=(np.asarray(np.where(dd == 14))).T
label15=(np.asarray(np.where(dd == 15))).T
label16=(np.asarray(np.where(dd == 16))).T
label17=(np.asarray(np.where(dd == 17))).T
label18=(np.asarray(np.where(dd == 18))).T
label19=(np.asarray(np.where(dd == 19))).T
label20=(np.asarray(np.where(dd == 20))).T
label21=(np.asarray(np.where(dd == 21))).T
label22=(np.asarray(np.where(dd == 22))).T
label23=(np.asarray(np.where(dd == 23))).T



print('set the output matrix')
clementanswer3=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,22),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,22),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,1),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,22),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,1),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,22),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,1),'F')
model3.fit(a3, b3)

##
model4 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a4=inputtrainclass[label4,:]
a4=np.reshape(a4,(-1,22),'F')

b4=ydami[label4,:]
b4=np.reshape(b4,(-1,1),'F')
model4.fit(a4, b4)

##
model5 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a5=inputtrainclass[label5,:]
a5=np.reshape(a5,(-1,22),'F')

b5=ydami[label5,:]
b5=np.reshape(b5,(-1,1),'F')
model5.fit(a5, b5)

##
model6 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a6=inputtrainclass[label6,:]
a6=np.reshape(a6,(-1,22),'F')

b6=ydami[label6,:]
b6=np.reshape(b6,(-1,1),'F')
model6.fit(a6, b6)

##
model7 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a7=inputtrainclass[label7,:]
a7=np.reshape(a7,(-1,22),'F')

b7=ydami[label7,:]
b7=np.reshape(b7,(-1,1),'F')
model7.fit(a7, b7)

##
model8 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a8=inputtrainclass[label8,:]
a8=np.reshape(a8,(-1,22),'F')

b8=ydami[label8,:]
b8=np.reshape(b8,(-1,1),'F')
model8.fit(a8, b8)

##
model9 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a9=inputtrainclass[label9,:]
a9=np.reshape(a9,(-1,22),'F')

b9=ydami[label9,:]
b9=np.reshape(b9,(-1,1),'F')
model9.fit(a9, b9)

##
model10 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a10=inputtrainclass[label10,:]
a10=np.reshape(a10,(-1,22),'F')

b10=ydami[label10,:]
b10=np.reshape(b10,(-1,1),'F')
model10.fit(a10, b10)

##
model11 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a11=inputtrainclass[label11,:]
a11=np.reshape(a11,(-1,22),'F')

b11=ydami[label11,:]
b11=np.reshape(b11,(-1,1),'F')
model11.fit(a11, b11)

##
model12 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a12=inputtrainclass[label12,:]
a12=np.reshape(a11,(-1,22),'F')

b12=ydami[label12,:]
b12=np.reshape(b12,(-1,1),'F')
model12.fit(a12, b12)


##
model13 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a13=inputtrainclass[label13,:]
a13=np.reshape(a13,(-1,22),'F')

b13=ydami[label13,:]
b13=np.reshape(b13,(-1,1),'F')
model13.fit(a13, b13)

##
model14 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a14=inputtrainclass[label14,:]
a14=np.reshape(a14,(-1,22),'F')

b14=ydami[label14,:]
b14=np.reshape(b14,(-1,1),'F')
model14.fit(a14, b14)

##
model15 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a15=inputtrainclass[label15,:]
a15=np.reshape(a15,(-1,22),'F')

b15=ydami[label15,:]
b15=np.reshape(b15,(-1,1),'F')
model15.fit(a15, b15)

##
model16 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a16=inputtrainclass[label16,:]
a16=np.reshape(a16,(-1,22),'F')

b16=ydami[label16,:]
b16=np.reshape(b16,(-1,1),'F')
model16.fit(a16, b16)

##
model17 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a17=inputtrainclass[label17,:]
a17=np.reshape(a17,(-1,22),'F')

b17=ydami[label17,:]
b17=np.reshape(b17,(-1,1),'F')
model17.fit(a17, b17)

##
model18 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a18=inputtrainclass[label18,:]
a18=np.reshape(a18,(-1,22),'F')

b18=ydami[label18,:]
b18=np.reshape(b18,(-1,1),'F')
model18.fit(a18, b18)

##
model19 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a19=inputtrainclass[label19,:]
a19=np.reshape(a19,(-1,22),'F')

b19=ydami[label19,:]
b19=np.reshape(b19,(-1,1),'F')
model19.fit(a19, b19)

##
model20 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a20=inputtrainclass[label20,:]
a20=np.reshape(a20,(-1,22),'F')

b20=ydami[label20,:]
b20=np.reshape(b20,(-1,1),'F')
model20.fit(a20, b20)

##
model21 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a21=inputtrainclass[label21,:]
a21=np.reshape(a21,(-1,22),'F')

b21=ydami[label21,:]
b21=np.reshape(b21,(-1,1),'F')
model21.fit(a21, b21)

##
model22 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a22=inputtrainclass[label22,:]
a22=np.reshape(a22,(-1,22),'F')

b22=ydami[label22,:]
b22=np.reshape(b22,(-1,1),'F')
model22.fit(a22, b22)

##
model23 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a23=inputtrainclass[label23,:]
a23=np.reshape(a23,(-1,22),'F')

b23=ydami[label23,:]
b23=np.reshape(b23,(-1,1),'F')
model23.fit(a23, b23)

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
labelDA10=(np.asarray(np.where(labelDA == 10))).T
labelDA11=(np.asarray(np.where(labelDA == 11))).T
labelDA12=(np.asarray(np.where(labelDA == 12))).T
labelDA13=(np.asarray(np.where(labelDA == 13))).T
labelDA14=(np.asarray(np.where(labelDA == 14))).T
labelDA15=(np.asarray(np.where(labelDA == 15))).T
labelDA16=(np.asarray(np.where(labelDA == 16))).T
labelDA17=(np.asarray(np.where(labelDA == 17))).T
labelDA18=(np.asarray(np.where(labelDA == 18))).T
labelDA19=(np.asarray(np.where(labelDA == 19))).T
labelDA20=(np.asarray(np.where(labelDA == 20))).T
labelDA21=(np.asarray(np.where(labelDA == 21))).T
labelDA22=(np.asarray(np.where(labelDA == 22))).T
labelDA23=(np.asarray(np.where(labelDA == 23))).T

##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,22),'F')
clementanswer3[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,22),'F')
clementanswer3[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,22),'F')
clementanswer3[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,22),'F')
clementanswer3[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))

a44=inputtest[labelDA4,:]
a44=np.reshape(a44,(-1,22),'F')
clementanswer1[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,1))

a55=inputtest[labelDA5,:]
a55=np.reshape(a55,(-1,22),'F')
clementanswer3[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,1))

a66=inputtest[labelDA6,:]
a66=np.reshape(a66,(-1,22),'F')
clementanswer3[np.ravel(labelDA6),:]=np.reshape(model6.predict(a66),(-1,1))

a77=inputtest[labelDA7,:]
a77=np.reshape(a77,(-1,22),'F')
clementanswer3[np.ravel(labelDA7),:]=np.reshape(model7.predict(a77),(-1,1))

a88=inputtest[labelDA8,:]
a88=np.reshape(a88,(-1,22),'F')
clementanswer1[np.ravel(labelDA8),:]=np.reshape(model8.predict(a88),(-1,1))

a99=inputtest[labelDA9,:]
a99=np.reshape(a99,(-1,22),'F')
clementanswer3[np.ravel(labelDA9),:]=np.reshape(model9.predict(a99),(-1,1))

a1010=inputtest[labelDA10,:]
a1010=np.reshape(a1010,(-1,22),'F')
clementanswer3[np.ravel(labelDA10),:]=np.reshape(model10.predict(a1010),(-1,1))

a1111=inputtest[labelDA11,:]
a1111=np.reshape(a1111,(-1,22),'F')
clementanswer3[np.ravel(labelDA11),:]=np.reshape(model11.predict(a1111),(-1,1))

a1212=inputtest[labelDA12,:]
a1212=np.reshape(a1212,(-1,22),'F')
clementanswer3[np.ravel(labelDA12),:]=np.reshape(model12.predict(a1212),(-1,1))

a1313=inputtest[labelDA13,:]
a1313=np.reshape(a1313,(-1,22),'F')
clementanswer3[np.ravel(labelDA13),:]=np.reshape(model13.predict(a1313),(-1,1))

a1414=inputtest[labelDA14,:]
a1414=np.reshape(a1414,(-1,22),'F')
clementanswer3[np.ravel(labelDA14),:]=np.reshape(model14.predict(a1414),(-1,1))

a1515=inputtest[labelDA15,:]
a1515=np.reshape(a1515,(-1,22),'F')
clementanswer3[np.ravel(labelDA15),:]=np.reshape(model15.predict(a1515),(-1,1))

a1616=inputtest[labelDA16,:]
a1616=np.reshape(a1616,(-1,22),'F')
clementanswer3[np.ravel(labelDA16),:]=np.reshape(model16.predict(a1616),(-1,1))

a1717=inputtest[labelDA17,:]
a1717=np.reshape(a1717,(-1,22),'F')
clementanswer3[np.ravel(labelDA17),:]=np.reshape(model17.predict(a1717),(-1,1))

a1818=inputtest[labelDA18,:]
a1818=np.reshape(a1818,(-1,22),'F')
clementanswer3[np.ravel(labelDA18),:]=np.reshape(model18.predict(a1818),(-1,1))

a1919=inputtest[labelDA19,:]
a1919=np.reshape(a1919,(-1,22),'F')
clementanswer3[np.ravel(labelDA19),:]=np.reshape(model19.predict(a1919),(-1,1))

a2020=inputtest[labelDA20,:]
a2020=np.reshape(a2020,(-1,22),'F')
clementanswer3[np.ravel(labelDA20),:]=np.reshape(model20.predict(a2020),(-1,1))

a2121=inputtest[labelDA21,:]
a2121=np.reshape(a2121,(-1,22),'F')
clementanswer3[np.ravel(labelDA21),:]=np.reshape(model21.predict(a2121),(-1,1))

a2222=inputtest[labelDA22,:]
a2222=np.reshape(a2222,(-1,22),'F')
clementanswer3[np.ravel(labelDA22),:]=np.reshape(model22.predict(a2222),(-1,1))

a2323=inputtest[labelDA23,:]
a2323=np.reshape(a2323,(-1,22),'F')
clementanswer3[np.ravel(labelDA23),:]=np.reshape(model23.predict(a2323),(-1,1))

print('For the 4th output')
outputtrain=np.reshape((output[0:300000,3]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,3]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((input11,np.reshape(output[:,3],(-1,1))), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=24,random_state=0,batch_size=6,max_iter=10).fit(matrix)
ddd=kmeans.labels_
ddd=ddd.T
dd=ddd[0:300000]
ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = RandomForestClassifier(n_estimators=100)
print('Predict the classes from the classifier for test data')
from sklearn.model_selection import train_test_split

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(inputtrainclass, outputtrainclass, test_size=0.999)

labelDAA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,inputtest,ddtest)
unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]

labelDA=queried_array[:,-1]
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
label10=(np.asarray(np.where(dd == 10))).T
label11=(np.asarray(np.where(dd == 11))).T
label12=(np.asarray(np.where(dd == 12))).T
label13=(np.asarray(np.where(dd == 13))).T
label14=(np.asarray(np.where(dd == 14))).T
label15=(np.asarray(np.where(dd == 15))).T
label16=(np.asarray(np.where(dd == 16))).T
label17=(np.asarray(np.where(dd == 17))).T
label18=(np.asarray(np.where(dd == 18))).T
label19=(np.asarray(np.where(dd == 19))).T
label20=(np.asarray(np.where(dd == 20))).T
label21=(np.asarray(np.where(dd == 21))).T
label22=(np.asarray(np.where(dd == 22))).T
label23=(np.asarray(np.where(dd == 23))).T



print('set the output matrix')
clementanswer4=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,22),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,22),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,1),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,22),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,1),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,22),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,1),'F')
model3.fit(a3, b3)

##
model4 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a4=inputtrainclass[label4,:]
a4=np.reshape(a4,(-1,22),'F')

b4=ydami[label4,:]
b4=np.reshape(b4,(-1,1),'F')
model4.fit(a4, b4)

##
model5 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a5=inputtrainclass[label5,:]
a5=np.reshape(a5,(-1,22),'F')

b5=ydami[label5,:]
b5=np.reshape(b5,(-1,1),'F')
model5.fit(a5, b5)

##
model6 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a6=inputtrainclass[label6,:]
a6=np.reshape(a6,(-1,22),'F')

b6=ydami[label6,:]
b6=np.reshape(b6,(-1,1),'F')
model6.fit(a6, b6)

##
model7 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a7=inputtrainclass[label7,:]
a7=np.reshape(a7,(-1,22),'F')

b7=ydami[label7,:]
b7=np.reshape(b7,(-1,1),'F')
model7.fit(a7, b7)

##
model8 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a8=inputtrainclass[label8,:]
a8=np.reshape(a8,(-1,22),'F')

b8=ydami[label8,:]
b8=np.reshape(b8,(-1,1),'F')
model8.fit(a8, b8)

##
model9 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a9=inputtrainclass[label9,:]
a9=np.reshape(a9,(-1,22),'F')

b9=ydami[label9,:]
b9=np.reshape(b9,(-1,1),'F')
model9.fit(a9, b9)

##
model10 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a10=inputtrainclass[label10,:]
a10=np.reshape(a10,(-1,22),'F')

b10=ydami[label10,:]
b10=np.reshape(b10,(-1,1),'F')
model10.fit(a10, b10)

##
model11 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a11=inputtrainclass[label11,:]
a11=np.reshape(a11,(-1,22),'F')

b11=ydami[label11,:]
b11=np.reshape(b11,(-1,1),'F')
model11.fit(a11, b11)

##
model12 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a12=inputtrainclass[label12,:]
a12=np.reshape(a11,(-1,22),'F')

b12=ydami[label12,:]
b12=np.reshape(b12,(-1,1),'F')
model12.fit(a12, b12)


##
model13 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a13=inputtrainclass[label13,:]
a13=np.reshape(a13,(-1,22),'F')

b13=ydami[label13,:]
b13=np.reshape(b13,(-1,1),'F')
model13.fit(a13, b13)

##
model14 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a14=inputtrainclass[label14,:]
a14=np.reshape(a14,(-1,22),'F')

b14=ydami[label14,:]
b14=np.reshape(b14,(-1,1),'F')
model14.fit(a14, b14)

##
model15 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a15=inputtrainclass[label15,:]
a15=np.reshape(a15,(-1,22),'F')

b15=ydami[label15,:]
b15=np.reshape(b15,(-1,1),'F')
model15.fit(a15, b15)

##
model16 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a16=inputtrainclass[label16,:]
a16=np.reshape(a16,(-1,22),'F')

b16=ydami[label16,:]
b16=np.reshape(b16,(-1,1),'F')
model16.fit(a16, b16)

##
model17 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a17=inputtrainclass[label17,:]
a17=np.reshape(a17,(-1,22),'F')

b17=ydami[label17,:]
b17=np.reshape(b17,(-1,1),'F')
model17.fit(a17, b17)

##
model18 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a18=inputtrainclass[label18,:]
a18=np.reshape(a18,(-1,22),'F')

b18=ydami[label18,:]
b18=np.reshape(b18,(-1,1),'F')
model18.fit(a18, b18)

##
model19 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a19=inputtrainclass[label19,:]
a19=np.reshape(a19,(-1,22),'F')

b19=ydami[label19,:]
b19=np.reshape(b19,(-1,1),'F')
model19.fit(a19, b19)

##
model20 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a20=inputtrainclass[label20,:]
a20=np.reshape(a20,(-1,22),'F')

b20=ydami[label20,:]
b20=np.reshape(b20,(-1,1),'F')
model20.fit(a20, b20)

##
model21 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a21=inputtrainclass[label21,:]
a21=np.reshape(a21,(-1,22),'F')

b21=ydami[label21,:]
b21=np.reshape(b21,(-1,1),'F')
model21.fit(a21, b21)

##
model22 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a22=inputtrainclass[label22,:]
a22=np.reshape(a22,(-1,22),'F')

b22=ydami[label22,:]
b22=np.reshape(b22,(-1,1),'F')
model22.fit(a22, b22)

##
model23 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a23=inputtrainclass[label23,:]
a23=np.reshape(a23,(-1,22),'F')

b23=ydami[label23,:]
b23=np.reshape(b23,(-1,1),'F')
model23.fit(a23, b23)

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
labelDA10=(np.asarray(np.where(labelDA == 10))).T
labelDA11=(np.asarray(np.where(labelDA == 11))).T
labelDA12=(np.asarray(np.where(labelDA == 12))).T
labelDA13=(np.asarray(np.where(labelDA == 13))).T
labelDA14=(np.asarray(np.where(labelDA == 14))).T
labelDA15=(np.asarray(np.where(labelDA == 15))).T
labelDA16=(np.asarray(np.where(labelDA == 16))).T
labelDA17=(np.asarray(np.where(labelDA == 17))).T
labelDA18=(np.asarray(np.where(labelDA == 18))).T
labelDA19=(np.asarray(np.where(labelDA == 19))).T
labelDA20=(np.asarray(np.where(labelDA == 20))).T
labelDA21=(np.asarray(np.where(labelDA == 21))).T
labelDA22=(np.asarray(np.where(labelDA == 22))).T
labelDA23=(np.asarray(np.where(labelDA == 23))).T

##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,22),'F')
clementanswer4[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,22),'F')
clementanswer4[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,22),'F')
clementanswer4[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,22),'F')
clementanswer4[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))

a44=inputtest[labelDA4,:]
a44=np.reshape(a44,(-1,22),'F')
clementanswer4[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,1))

a55=inputtest[labelDA5,:]
a55=np.reshape(a55,(-1,22),'F')
clementanswer4[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,1))

a66=inputtest[labelDA6,:]
a66=np.reshape(a66,(-1,22),'F')
clementanswer4[np.ravel(labelDA6),:]=np.reshape(model6.predict(a66),(-1,1))

a77=inputtest[labelDA7,:]
a77=np.reshape(a77,(-1,22),'F')
clementanswer4[np.ravel(labelDA7),:]=np.reshape(model7.predict(a77),(-1,1))

a88=inputtest[labelDA8,:]
a88=np.reshape(a88,(-1,22),'F')
clementanswer4[np.ravel(labelDA8),:]=np.reshape(model8.predict(a88),(-1,1))

a99=inputtest[labelDA9,:]
a99=np.reshape(a99,(-1,22),'F')
clementanswer4[np.ravel(labelDA9),:]=np.reshape(model9.predict(a99),(-1,1))

a1010=inputtest[labelDA10,:]
a1010=np.reshape(a1010,(-1,22),'F')
clementanswer1[np.ravel(labelDA10),:]=np.reshape(model10.predict(a1010),(-1,1))

a1111=inputtest[labelDA11,:]
a1111=np.reshape(a1111,(-1,22),'F')
clementanswer4[np.ravel(labelDA11),:]=np.reshape(model11.predict(a1111),(-1,1))

a1212=inputtest[labelDA12,:]
a1212=np.reshape(a1212,(-1,22),'F')
clementanswer4[np.ravel(labelDA12),:]=np.reshape(model12.predict(a1212),(-1,1))

a1313=inputtest[labelDA13,:]
a1313=np.reshape(a1313,(-1,22),'F')
clementanswer4[np.ravel(labelDA13),:]=np.reshape(model13.predict(a1313),(-1,1))

a1414=inputtest[labelDA14,:]
a1414=np.reshape(a1414,(-1,22),'F')
clementanswer4[np.ravel(labelDA14),:]=np.reshape(model14.predict(a1414),(-1,1))

a1515=inputtest[labelDA15,:]
a1515=np.reshape(a1515,(-1,22),'F')
clementanswer4[np.ravel(labelDA15),:]=np.reshape(model15.predict(a1515),(-1,1))

a1616=inputtest[labelDA16,:]
a1616=np.reshape(a1616,(-1,22),'F')
clementanswer4[np.ravel(labelDA16),:]=np.reshape(model16.predict(a1616),(-1,1))

a1717=inputtest[labelDA17,:]
a1717=np.reshape(a1717,(-1,22),'F')
clementanswer4[np.ravel(labelDA17),:]=np.reshape(model17.predict(a1717),(-1,1))

a1818=inputtest[labelDA18,:]
a1818=np.reshape(a1818,(-1,22),'F')
clementanswer4[np.ravel(labelDA18),:]=np.reshape(model18.predict(a1818),(-1,1))

a1919=inputtest[labelDA19,:]
a1919=np.reshape(a1919,(-1,22),'F')
clementanswer4[np.ravel(labelDA19),:]=np.reshape(model19.predict(a1919),(-1,1))

a2020=inputtest[labelDA20,:]
a2020=np.reshape(a2020,(-1,22),'F')
clementanswer4[np.ravel(labelDA20),:]=np.reshape(model20.predict(a2020),(-1,1))

a2121=inputtest[labelDA21,:]
a2121=np.reshape(a2121,(-1,22),'F')
clementanswer4[np.ravel(labelDA21),:]=np.reshape(model21.predict(a2121),(-1,1))

a2222=inputtest[labelDA22,:]
a2222=np.reshape(a2222,(-1,22),'F')
clementanswer4[np.ravel(labelDA22),:]=np.reshape(model22.predict(a2222),(-1,1))

a2323=inputtest[labelDA23,:]
a2323=np.reshape(a2323,(-1,22),'F')
clementanswer4[np.ravel(labelDA23),:]=np.reshape(model23.predict(a2323),(-1,1))

print('For the 5th output')

outputtrain=np.reshape((output[0:300000,4]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,4]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((input11,np.reshape(output[:,4],(-1,1))), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=24,random_state=0,batch_size=6,max_iter=10).fit(matrix)
ddd=kmeans.labels_
ddd=ddd.T
dd=ddd[0:300000]
ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = RandomForestClassifier(n_estimators=100)
print('Predict the classes from the classifier for test data')
from sklearn.model_selection import train_test_split

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(inputtrainclass, outputtrainclass, test_size=0.999)

labelDAA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,inputtest,ddtest)
unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]

labelDA=queried_array[:,-1]
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
label10=(np.asarray(np.where(dd == 10))).T
label11=(np.asarray(np.where(dd == 11))).T
label12=(np.asarray(np.where(dd == 12))).T
label13=(np.asarray(np.where(dd == 13))).T
label14=(np.asarray(np.where(dd == 14))).T
label15=(np.asarray(np.where(dd == 15))).T
label16=(np.asarray(np.where(dd == 16))).T
label17=(np.asarray(np.where(dd == 17))).T
label18=(np.asarray(np.where(dd == 18))).T
label19=(np.asarray(np.where(dd == 19))).T
label20=(np.asarray(np.where(dd == 20))).T
label21=(np.asarray(np.where(dd == 21))).T
label22=(np.asarray(np.where(dd == 22))).T
label23=(np.asarray(np.where(dd == 23))).T



print('set the output matrix')
clementanswer5=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,22),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,22),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,1),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,22),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,1),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,22),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,1),'F')
model3.fit(a3, b3)

##
model4 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a4=inputtrainclass[label4,:]
a4=np.reshape(a4,(-1,22),'F')

b4=ydami[label4,:]
b4=np.reshape(b4,(-1,1),'F')
model4.fit(a4, b4)

##
model5 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a5=inputtrainclass[label5,:]
a5=np.reshape(a5,(-1,22),'F')

b5=ydami[label5,:]
b5=np.reshape(b5,(-1,1),'F')
model5.fit(a5, b5)

##
model6 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a6=inputtrainclass[label6,:]
a6=np.reshape(a6,(-1,22),'F')

b6=ydami[label6,:]
b6=np.reshape(b6,(-1,1),'F')
model6.fit(a6, b6)

##
model7 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a7=inputtrainclass[label7,:]
a7=np.reshape(a7,(-1,22),'F')

b7=ydami[label7,:]
b7=np.reshape(b7,(-1,1),'F')
model7.fit(a7, b7)

##
model8 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a8=inputtrainclass[label8,:]
a8=np.reshape(a8,(-1,22),'F')

b8=ydami[label8,:]
b8=np.reshape(b8,(-1,1),'F')
model8.fit(a8, b8)

##
model9 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a9=inputtrainclass[label9,:]
a9=np.reshape(a9,(-1,22),'F')

b9=ydami[label9,:]
b9=np.reshape(b9,(-1,1),'F')
model9.fit(a9, b9)

##
model10 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a10=inputtrainclass[label10,:]
a10=np.reshape(a10,(-1,22),'F')

b10=ydami[label10,:]
b10=np.reshape(b10,(-1,1),'F')
model10.fit(a10, b10)

##
model11 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a11=inputtrainclass[label11,:]
a11=np.reshape(a11,(-1,22),'F')

b11=ydami[label11,:]
b11=np.reshape(b11,(-1,1),'F')
model11.fit(a11, b11)

##
model12 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a12=inputtrainclass[label12,:]
a12=np.reshape(a11,(-1,22),'F')

b12=ydami[label12,:]
b12=np.reshape(b12,(-1,1),'F')
model12.fit(a12, b12)


##
model13 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a13=inputtrainclass[label13,:]
a13=np.reshape(a13,(-1,22),'F')

b13=ydami[label13,:]
b13=np.reshape(b13,(-1,1),'F')
model13.fit(a13, b13)

##
model14 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a14=inputtrainclass[label14,:]
a14=np.reshape(a14,(-1,22),'F')

b14=ydami[label14,:]
b14=np.reshape(b14,(-1,1),'F')
model14.fit(a14, b14)

##
model15 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a15=inputtrainclass[label15,:]
a15=np.reshape(a15,(-1,22),'F')

b15=ydami[label15,:]
b15=np.reshape(b15,(-1,1),'F')
model15.fit(a15, b15)

##
model16 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a16=inputtrainclass[label16,:]
a16=np.reshape(a16,(-1,22),'F')

b16=ydami[label16,:]
b16=np.reshape(b16,(-1,1),'F')
model16.fit(a16, b16)

##
model17 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a17=inputtrainclass[label17,:]
a17=np.reshape(a17,(-1,22),'F')

b17=ydami[label17,:]
b17=np.reshape(b17,(-1,1),'F')
model17.fit(a17, b17)

##
model18 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a18=inputtrainclass[label18,:]
a18=np.reshape(a18,(-1,22),'F')

b18=ydami[label18,:]
b18=np.reshape(b18,(-1,1),'F')
model18.fit(a18, b18)

##
model19 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a19=inputtrainclass[label19,:]
a19=np.reshape(a19,(-1,22),'F')

b19=ydami[label19,:]
b19=np.reshape(b19,(-1,1),'F')
model19.fit(a19, b19)

##
model20 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a20=inputtrainclass[label20,:]
a20=np.reshape(a20,(-1,22),'F')

b20=ydami[label20,:]
b20=np.reshape(b20,(-1,1),'F')
model20.fit(a20, b20)

##
model21 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a21=inputtrainclass[label21,:]
a21=np.reshape(a21,(-1,22),'F')

b21=ydami[label21,:]
b21=np.reshape(b21,(-1,1),'F')
model21.fit(a21, b21)

##
model22 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a22=inputtrainclass[label22,:]
a22=np.reshape(a22,(-1,22),'F')

b22=ydami[label22,:]
b22=np.reshape(b22,(-1,1),'F')
model22.fit(a22, b22)

##
model23 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a23=inputtrainclass[label23,:]
a23=np.reshape(a23,(-1,22),'F')

b23=ydami[label23,:]
b23=np.reshape(b23,(-1,1),'F')
model23.fit(a23, b23)

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
labelDA10=(np.asarray(np.where(labelDA == 10))).T
labelDA11=(np.asarray(np.where(labelDA == 11))).T
labelDA12=(np.asarray(np.where(labelDA == 12))).T
labelDA13=(np.asarray(np.where(labelDA == 13))).T
labelDA14=(np.asarray(np.where(labelDA == 14))).T
labelDA15=(np.asarray(np.where(labelDA == 15))).T
labelDA16=(np.asarray(np.where(labelDA == 16))).T
labelDA17=(np.asarray(np.where(labelDA == 17))).T
labelDA18=(np.asarray(np.where(labelDA == 18))).T
labelDA19=(np.asarray(np.where(labelDA == 19))).T
labelDA20=(np.asarray(np.where(labelDA == 20))).T
labelDA21=(np.asarray(np.where(labelDA == 21))).T
labelDA22=(np.asarray(np.where(labelDA == 22))).T
labelDA23=(np.asarray(np.where(labelDA == 23))).T

##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,22),'F')
clementanswer5[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,22),'F')
clementanswer5[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,22),'F')
clementanswer5[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,22),'F')
clementanswer5[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))

a44=inputtest[labelDA4,:]
a44=np.reshape(a44,(-1,22),'F')
clementanswer1[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,1))

a55=inputtest[labelDA5,:]
a55=np.reshape(a55,(-1,22),'F')
clementanswer1[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,1))

a66=inputtest[labelDA6,:]
a66=np.reshape(a66,(-1,22),'F')
clementanswer5[np.ravel(labelDA6),:]=np.reshape(model6.predict(a66),(-1,1))

a77=inputtest[labelDA7,:]
a77=np.reshape(a77,(-1,22),'F')
clementanswer5[np.ravel(labelDA7),:]=np.reshape(model7.predict(a77),(-1,1))

a88=inputtest[labelDA8,:]
a88=np.reshape(a88,(-1,22),'F')
clementanswer5[np.ravel(labelDA8),:]=np.reshape(model8.predict(a88),(-1,1))

a99=inputtest[labelDA9,:]
a99=np.reshape(a99,(-1,22),'F')
clementanswer1[np.ravel(labelDA9),:]=np.reshape(model9.predict(a99),(-1,1))

a1010=inputtest[labelDA10,:]
a1010=np.reshape(a1010,(-1,22),'F')
clementanswer5[np.ravel(labelDA10),:]=np.reshape(model10.predict(a1010),(-1,1))

a1111=inputtest[labelDA11,:]
a1111=np.reshape(a1111,(-1,22),'F')
clementanswer1[np.ravel(labelDA11),:]=np.reshape(model11.predict(a1111),(-1,1))

a1212=inputtest[labelDA12,:]
a1212=np.reshape(a1212,(-1,22),'F')
clementanswer5[np.ravel(labelDA12),:]=np.reshape(model12.predict(a1212),(-1,1))

a1313=inputtest[labelDA13,:]
a1313=np.reshape(a1313,(-1,22),'F')
clementanswer1[np.ravel(labelDA13),:]=np.reshape(model13.predict(a1313),(-1,1))

a1414=inputtest[labelDA14,:]
a1414=np.reshape(a1414,(-1,22),'F')
clementanswer5[np.ravel(labelDA14),:]=np.reshape(model14.predict(a1414),(-1,1))

a1515=inputtest[labelDA15,:]
a1515=np.reshape(a1515,(-1,22),'F')
clementanswer5[np.ravel(labelDA15),:]=np.reshape(model15.predict(a1515),(-1,1))

a1616=inputtest[labelDA16,:]
a1616=np.reshape(a1616,(-1,22),'F')
clementanswer5[np.ravel(labelDA16),:]=np.reshape(model16.predict(a1616),(-1,1))

a1717=inputtest[labelDA17,:]
a1717=np.reshape(a1717,(-1,22),'F')
clementanswer1[np.ravel(labelDA17),:]=np.reshape(model17.predict(a1717),(-1,1))

a1818=inputtest[labelDA18,:]
a1818=np.reshape(a1818,(-1,22),'F')
clementanswer5[np.ravel(labelDA18),:]=np.reshape(model18.predict(a1818),(-1,1))

a1919=inputtest[labelDA19,:]
a1919=np.reshape(a1919,(-1,22),'F')
clementanswer1[np.ravel(labelDA19),:]=np.reshape(model19.predict(a1919),(-1,1))

a2020=inputtest[labelDA20,:]
a2020=np.reshape(a2020,(-1,22),'F')
clementanswer5[np.ravel(labelDA20),:]=np.reshape(model20.predict(a2020),(-1,1))

a2121=inputtest[labelDA21,:]
a2121=np.reshape(a2121,(-1,22),'F')
clementanswer5[np.ravel(labelDA21),:]=np.reshape(model21.predict(a2121),(-1,1))

a2222=inputtest[labelDA22,:]
a2222=np.reshape(a2222,(-1,22),'F')
clementanswer5[np.ravel(labelDA22),:]=np.reshape(model22.predict(a2222),(-1,1))

a2323=inputtest[labelDA23,:]
a2323=np.reshape(a2323,(-1,22),'F')
clementanswer5[np.ravel(labelDA23),:]=np.reshape(model23.predict(a2323),(-1,1))

print('fot the 6th output')

outputtrain=np.reshape((output[0:300000,5]),(-1,1)) #select the first 300000 for training
ydami=outputtrain;
#outputtrain=np.arcsinh(outputtrain)
outputtest=np.reshape((output[300001:numrows,5]),(-1,1))
numrowstest=len(outputtest)    # rows of inout
numcolstest = len(outputtest[0])

matrix=np.concatenate((input11,np.reshape(output[:,5],(-1,1))), axis=1)

print('Do the K-means clustering with 10 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=24,random_state=0,batch_size=6,max_iter=10).fit(matrix)
ddd=kmeans.labels_
ddd=ddd.T
dd=ddd[0:300000]
ddtest=ddd[300001:numrows]
dd=np.reshape(dd,(-1,1))
ddtest=np.reshape(ddtest,(-1,1))
#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=inputtrain
outputtrainclass=np.reshape(dd,(-1,1))
from sklearn.ensemble import RandomForestClassifier
print(' Learn the classifer from the predicted labels from Kmeans')
#model = MLPClassifier(learning_rate='adaptive',hidden_layer_sizes=(200, ),solver= 'lbfgs',max_iter=3000,validation_fraction=0.05)
model = RandomForestClassifier(n_estimators=100)
print('Predict the classes from the classifier for test data')
from sklearn.model_selection import train_test_split

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(inputtrainclass, outputtrainclass, test_size=0.999)

labelDAA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,inputtest,ddtest)
unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]

labelDA=queried_array[:,-1]
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
label10=(np.asarray(np.where(dd == 10))).T
label11=(np.asarray(np.where(dd == 11))).T
label12=(np.asarray(np.where(dd == 12))).T
label13=(np.asarray(np.where(dd == 13))).T
label14=(np.asarray(np.where(dd == 14))).T
label15=(np.asarray(np.where(dd == 15))).T
label16=(np.asarray(np.where(dd == 16))).T
label17=(np.asarray(np.where(dd == 17))).T
label18=(np.asarray(np.where(dd == 18))).T
label19=(np.asarray(np.where(dd == 19))).T
label20=(np.asarray(np.where(dd == 20))).T
label21=(np.asarray(np.where(dd == 21))).T
label22=(np.asarray(np.where(dd == 22))).T
label23=(np.asarray(np.where(dd == 23))).T



print('set the output matrix')
clementanswer6=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
##
model0 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a0=inputtrainclass[label0,:]
a0=np.reshape(a0,(-1,22),'F')

b0=ydami[label0,:]
b0=np.reshape(b0,(-1,1),'F')
model0.fit(a0, b0)

##
model1 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a1=inputtrainclass[label1,:]
a1=np.reshape(a1,(-1,22),'F')

b1=ydami[label1,:]
b1=np.reshape(b1,(-1,1),'F')
model1.fit(a1, b1)

##
model2 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a2=inputtrainclass[label2,:]
a2=np.reshape(a2,(-1,22),'F')

b2=ydami[label2,:]
b2=np.reshape(b2,(-1,1),'F')
model2.fit(a2, b2)

##
model3 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a3=inputtrainclass[label3,:]
a3=np.reshape(a3,(-1,22),'F')

b3=ydami[label3,:]
b3=np.reshape(b3,(-1,1),'F')
model3.fit(a3, b3)

##
model4 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a4=inputtrainclass[label4,:]
a4=np.reshape(a4,(-1,22),'F')

b4=ydami[label4,:]
b4=np.reshape(b4,(-1,1),'F')
model4.fit(a4, b4)

##
model5 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a5=inputtrainclass[label5,:]
a5=np.reshape(a5,(-1,22),'F')

b5=ydami[label5,:]
b5=np.reshape(b5,(-1,1),'F')
model5.fit(a5, b5)

##
model6 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a6=inputtrainclass[label6,:]
a6=np.reshape(a6,(-1,22),'F')

b6=ydami[label6,:]
b6=np.reshape(b6,(-1,1),'F')
model6.fit(a6, b6)

##
model7 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a7=inputtrainclass[label7,:]
a7=np.reshape(a7,(-1,22),'F')

b7=ydami[label7,:]
b7=np.reshape(b7,(-1,1),'F')
model7.fit(a7, b7)

##
model8 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a8=inputtrainclass[label8,:]
a8=np.reshape(a8,(-1,22),'F')

b8=ydami[label8,:]
b8=np.reshape(b8,(-1,1),'F')
model8.fit(a8, b8)

##
model9 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a9=inputtrainclass[label9,:]
a9=np.reshape(a9,(-1,22),'F')

b9=ydami[label9,:]
b9=np.reshape(b9,(-1,1),'F')
model9.fit(a9, b9)

##
model10 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a10=inputtrainclass[label10,:]
a10=np.reshape(a10,(-1,22),'F')

b10=ydami[label10,:]
b10=np.reshape(b10,(-1,1),'F')
model10.fit(a10, b10)

##
model11 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a11=inputtrainclass[label11,:]
a11=np.reshape(a11,(-1,22),'F')

b11=ydami[label11,:]
b11=np.reshape(b11,(-1,1),'F')
model11.fit(a11, b11)

##
model12 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a12=inputtrainclass[label12,:]
a12=np.reshape(a11,(-1,22),'F')

b12=ydami[label12,:]
b12=np.reshape(b12,(-1,1),'F')
model12.fit(a12, b12)


##
model13 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a13=inputtrainclass[label13,:]
a13=np.reshape(a13,(-1,22),'F')

b13=ydami[label13,:]
b13=np.reshape(b13,(-1,1),'F')
model13.fit(a13, b13)

##
model14 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a14=inputtrainclass[label14,:]
a14=np.reshape(a14,(-1,22),'F')

b14=ydami[label14,:]
b14=np.reshape(b14,(-1,1),'F')
model14.fit(a14, b14)

##
model15 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a15=inputtrainclass[label15,:]
a15=np.reshape(a15,(-1,22),'F')

b15=ydami[label15,:]
b15=np.reshape(b15,(-1,1),'F')
model15.fit(a15, b15)

##
model16 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a16=inputtrainclass[label16,:]
a16=np.reshape(a16,(-1,22),'F')

b16=ydami[label16,:]
b16=np.reshape(b16,(-1,1),'F')
model16.fit(a16, b16)

##
model17 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a17=inputtrainclass[label17,:]
a17=np.reshape(a17,(-1,22),'F')

b17=ydami[label17,:]
b17=np.reshape(b17,(-1,1),'F')
model17.fit(a17, b17)

##
model18 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a18=inputtrainclass[label18,:]
a18=np.reshape(a18,(-1,22),'F')

b18=ydami[label18,:]
b18=np.reshape(b18,(-1,1),'F')
model18.fit(a18, b18)

##
model19 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a19=inputtrainclass[label19,:]
a19=np.reshape(a19,(-1,22),'F')

b19=ydami[label19,:]
b19=np.reshape(b19,(-1,1),'F')
model19.fit(a19, b19)

##
model20 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a20=inputtrainclass[label20,:]
a20=np.reshape(a20,(-1,22),'F')

b20=ydami[label20,:]
b20=np.reshape(b20,(-1,1),'F')
model20.fit(a20, b20)

##
model21 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a21=inputtrainclass[label21,:]
a21=np.reshape(a21,(-1,22),'F')

b21=ydami[label21,:]
b21=np.reshape(b21,(-1,1),'F')
model21.fit(a21, b21)

##
model22 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a22=inputtrainclass[label22,:]
a22=np.reshape(a22,(-1,22),'F')

b22=ydami[label22,:]
b22=np.reshape(b22,(-1,1),'F')
model22.fit(a22, b22)

##
model23 = MLPRegressor(solver= 'lbfgs',max_iter=3000)
a23=inputtrainclass[label23,:]
a23=np.reshape(a23,(-1,22),'F')

b23=ydami[label23,:]
b23=np.reshape(b23,(-1,1),'F')
model23.fit(a23, b23)

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
labelDA10=(np.asarray(np.where(labelDA == 10))).T
labelDA11=(np.asarray(np.where(labelDA == 11))).T
labelDA12=(np.asarray(np.where(labelDA == 12))).T
labelDA13=(np.asarray(np.where(labelDA == 13))).T
labelDA14=(np.asarray(np.where(labelDA == 14))).T
labelDA15=(np.asarray(np.where(labelDA == 15))).T
labelDA16=(np.asarray(np.where(labelDA == 16))).T
labelDA17=(np.asarray(np.where(labelDA == 17))).T
labelDA18=(np.asarray(np.where(labelDA == 18))).T
labelDA19=(np.asarray(np.where(labelDA == 19))).T
labelDA20=(np.asarray(np.where(labelDA == 20))).T
labelDA21=(np.asarray(np.where(labelDA == 21))).T
labelDA22=(np.asarray(np.where(labelDA == 22))).T
labelDA23=(np.asarray(np.where(labelDA == 23))).T

##----------------------##------------------------##
a00=inputtest[labelDA0,:]
a00=np.reshape(a00,(-1,22),'F')
clementanswer6[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))


a11=inputtest[labelDA1,:]
a11=np.reshape(a11,(-1,22),'F')
clementanswer6[np.ravel(labelDA1),:]=np.reshape(model1.predict(a11),(-1,1))

a22=inputtest[labelDA2,:]
a22=np.reshape(a22,(-1,22),'F')
clementanswer6[np.ravel(labelDA2),:]=np.reshape(model2.predict(a22),(-1,1))

a33=inputtest[labelDA3,:]
a33=np.reshape(a33,(-1,22),'F')
clementanswer6[np.ravel(labelDA3),:]=np.reshape(model3.predict(a33),(-1,1))

a44=inputtest[labelDA4,:]
a44=np.reshape(a44,(-1,22),'F')
clementanswer6[np.ravel(labelDA4),:]=np.reshape(model4.predict(a44),(-1,1))

a55=inputtest[labelDA5,:]
a55=np.reshape(a55,(-1,22),'F')
clementanswer6[np.ravel(labelDA5),:]=np.reshape(model5.predict(a55),(-1,1))

a66=inputtest[labelDA6,:]
a66=np.reshape(a66,(-1,22),'F')
clementanswer6[np.ravel(labelDA6),:]=np.reshape(model6.predict(a66),(-1,1))

a77=inputtest[labelDA7,:]
a77=np.reshape(a77,(-1,22),'F')
clementanswer6[np.ravel(labelDA7),:]=np.reshape(model7.predict(a77),(-1,1))

a88=inputtest[labelDA8,:]
a88=np.reshape(a88,(-1,22),'F')
clementanswer6[np.ravel(labelDA8),:]=np.reshape(model8.predict(a88),(-1,1))

a99=inputtest[labelDA9,:]
a99=np.reshape(a99,(-1,22),'F')
clementanswer6[np.ravel(labelDA9),:]=np.reshape(model9.predict(a99),(-1,1))

a1010=inputtest[labelDA10,:]
a1010=np.reshape(a1010,(-1,22),'F')
clementanswer6[np.ravel(labelDA10),:]=np.reshape(model10.predict(a1010),(-1,1))

a1111=inputtest[labelDA11,:]
a1111=np.reshape(a1111,(-1,22),'F')
clementanswer6[np.ravel(labelDA11),:]=np.reshape(model11.predict(a1111),(-1,1))

a1212=inputtest[labelDA12,:]
a1212=np.reshape(a1212,(-1,22),'F')
clementanswer6[np.ravel(labelDA12),:]=np.reshape(model12.predict(a1212),(-1,1))

a1313=inputtest[labelDA13,:]
a1313=np.reshape(a1313,(-1,22),'F')
clementanswer6[np.ravel(labelDA13),:]=np.reshape(model13.predict(a1313),(-1,1))

a1414=inputtest[labelDA14,:]
a1414=np.reshape(a1414,(-1,22),'F')
clementanswer6[np.ravel(labelDA14),:]=np.reshape(model14.predict(a1414),(-1,1))

a1515=inputtest[labelDA15,:]
a1515=np.reshape(a1515,(-1,22),'F')
clementanswer6[np.ravel(labelDA15),:]=np.reshape(model15.predict(a1515),(-1,1))

a1616=inputtest[labelDA16,:]
a1616=np.reshape(a1616,(-1,22),'F')
clementanswer6[np.ravel(labelDA16),:]=np.reshape(model16.predict(a1616),(-1,1))

a1717=inputtest[labelDA17,:]
a1717=np.reshape(a1717,(-1,22),'F')
clementanswer6[np.ravel(labelDA17),:]=np.reshape(model17.predict(a1717),(-1,1))

a1818=inputtest[labelDA18,:]
a1818=np.reshape(a1818,(-1,22),'F')
clementanswer6[np.ravel(labelDA18),:]=np.reshape(model18.predict(a1818),(-1,1))

a1919=inputtest[labelDA19,:]
a1919=np.reshape(a1919,(-1,22),'F')
clementanswer6[np.ravel(labelDA19),:]=np.reshape(model19.predict(a1919),(-1,1))

a2020=inputtest[labelDA20,:]
a2020=np.reshape(a2020,(-1,22),'F')
clementanswer6[np.ravel(labelDA20),:]=np.reshape(model20.predict(a2020),(-1,1))

a2121=inputtest[labelDA21,:]
a2121=np.reshape(a2121,(-1,22),'F')
clementanswer6[np.ravel(labelDA21),:]=np.reshape(model21.predict(a2121),(-1,1))

a2222=inputtest[labelDA22,:]
a2222=np.reshape(a2222,(-1,22),'F')
clementanswer6[np.ravel(labelDA22),:]=np.reshape(model22.predict(a2222),(-1,1))

a2323=inputtest[labelDA23,:]
a2323=np.reshape(a2323,(-1,22),'F')
clementanswer6[np.ravel(labelDA23),:]=np.reshape(model23.predict(a2323),(-1,1))

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
plt.plot(outputteste[0:10000,0], color = 'red', label = 'Real data')
plt.plot(clementanswer1[0:10000,:], color = 'blue', label = 'Predicted data from Machine')

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
plt.plot(outputteste[0:10000,1], color = 'red', label = 'Real data')
plt.plot(clementanswer2[0:10000,:], color = 'blue', label = 'Predicted data from Machine')

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
plt.plot(outputteste[0:10000,2], color = 'red', label = 'Real data')
plt.plot(clementanswer3[0:10000,:], color = 'blue', label = 'Predicted data from Machine')

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
plt.plot(outputteste[0:10000,3], color = 'red', label = 'Real data')
plt.plot(clementanswer4[0:10000,:], color = 'blue', label = 'Predicted data from Machine')

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
plt.plot(outputteste[0:10000,4], color = 'red', label = 'Real data')
plt.plot(clementanswer5[0:10000,:], color = 'blue', label = 'Predicted data from Machine')

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
plt.plot(outputteste[0:10000,5], color = 'red', label = 'Real data')
plt.plot(clementanswer6[0:10000,:], color = 'blue', label = 'Predicted data from Machine')

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
plt.scatter(clementanswer1[0:10000,:],outputteste[0:10000,0], color ='c')
plt.title('output 1', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])

plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,2)
plt.scatter(clementanswer2[0:10000,:],outputteste[0:10000,1], color ='c')
plt.title('output 2', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()



fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,3)
plt.scatter(clementanswer3[0:10000,:],outputteste[0:10000,2], color ='c')
plt.title('output 3', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()




fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,4)
plt.scatter(clementanswer4[0:10000,:],outputteste[0:10000,3], color ='c')
plt.title('output 4', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()

fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,5)
plt.scatter(clementanswer5[0:10000,:],outputteste[0:10000,4], color ='c')
plt.title('output 5', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()


fig1 = plt.figure(figsize =(8,8))
fig1.add_subplot(3,2,6)
plt.scatter(clementanswer6[0:10000,:],outputteste[0:10000,5], color ='c')
plt.title('output 6', fontsize = 15)
plt.ylabel('Machine',fontsize = 13)
plt.xlabel('True data',fontsize = 13)
plt.axis([0,(100),0,(100)])
plt.show()

print('end of program')


