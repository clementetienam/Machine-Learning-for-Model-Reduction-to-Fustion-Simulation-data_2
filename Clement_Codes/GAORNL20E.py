# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 21 12:05:47 2019

@author: Dr Clement Etienam
This is the code for learning a machine for discountinous Chi function
We will cluster th data first, use that label from the cluster and learn a
classifier then a regressor
This code is very important for Chi-data infused with Active learning
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
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix
#----------------------Active Learning Module--------------------#
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
    
    
    
    
    max_queried=6000
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
            
            filename = 'finalizedclasschi_model.sav' #Save the classification model
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
#------------------Begin Code----------------#
start_time = datetime.now()
x = np.genfromtxt("chi_itg.dat",skip_header=1, dtype='float')
nclusters = np.int(input("Enter the number of clusters you want: ") )
print('cluster with X and y')
df=x
test=df
numrows=len(test)    # rows of inout
X=np.log(df[:,0:10])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
(scaler.fit(X))
X=(scaler.transform(X))
y=test[:,-1]
y=np.reshape(y,(-1,1));
ydami=np.reshape(y[0:290000],(-1,1));
outputtest=y[290000:600001]
numrowstest=len(outputtest)
inputtrainclass=X[0:290000,:]
#
outputtrainclass=np.reshape(y[0:290000],(-1,1))
matrix=np.concatenate((X,y), axis=1)
inputtest=X[290000:numrows,:]
print('Do the K-means clustering with 5 clusters of [X,y] and get the labels')
kmeans = MiniBatchKMeans(n_clusters=nclusters,random_state=0,batch_size=6,max_iter=10).fit(matrix)
ddd=kmeans.labels_
ddd=ddd.T
dd=ddd[0:290000]
ddtest=ddd[290000:numrows]
dd=np.reshape(dd,(-1,1))
ddtest=np.reshape(ddtest,(-1,1))

#-------------------#---------------------------------#
print('Use the labels to train a classifier')
inputtrainclass=X[0:290000,:]
outputtrainclass=np.reshape(dd,(-1,1))

inputtest=X[290000:numrows,:];
#%outputtest=y(290000+1:end,:);
#def run_model(model):
#    # build the model on training data
#    model.fit(inputtrainclass, outputtrainclass )
#
#    # make predictions for test data
#    labelDA = model.predict(inputtest)
#    return labelDA
print(' Learn the classifer from the predicted labels from Kmeans')
model = MLPClassifier(solver= 'lbfgs',max_iter=3000)
#model = RandomForestClassifier(n_estimators=500)
from sklearn.model_selection import train_test_split
print('Predict the classes from the classifier for test data')
X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(inputtrainclass, outputtrainclass, test_size=0.999)

labelDAA,ff_array,queried_array,queried_array2=run_model(model,X_labeled,y_labeled,X_unlabeled,y_oracle,inputtest,ddtest)
unie=np.amax(ff_array)
ff_array=np.reshape(ff_array,(-1,1))

label0=(np.asarray(np.where(ff_array==unie))).T
finallabel=queried_array[:,label0[0,0]]

labelDA=queried_array[:,-1]


 
# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(inputtest, ddtest)
#-------------------Regression----------------#
print('set the output matrix')
clementanswer=np.zeros((numrowstest,1))

print('Start the regression')
from sklearn.neural_network import MLPRegressor
print('Learn regression of the 5 clusters with different labels from k-means ' )
for i in range(nclusters):
    labela=(np.asarray(np.where(dd == i))).T
#label1=(np.asarray(np.where(dd == 1))).T
#label2=(np.asarray(np.where(dd == 2))).T
#label3=(np.asarray(np.where(dd == 3))).T
#label4=(np.asarray(np.where(dd == 4))).T


##
    modela = MLPRegressor(solver= 'lbfgs',max_iter=3000)
    a0=inputtrainclass[labela,:]
    a0=np.reshape(a0,(-1,10),'F')

    b0=ydami[labela,:]
    b0=np.reshape(b0,(-1,1),'F')
    modela.fit(a0, b0)



    print('Time for the prediction')
    labelDA0=(np.asarray(np.where(labelDA == i))).T



    a00=inputtest[labelDA0,:]
    a00=np.reshape(a00,(-1,10),'F')
    if a00.shape[0]!=0:
       clementanswer[np.ravel(labelDA0),:]=np.reshape(modela.predict(a00),(-1,1))


print(' Compute L2 and R2 for the machine')

outputtest = np.reshape(outputtest, (numrowstest, 1))
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
fig1.add_subplot(2,2,1)
plt.scatter(clementanswer[0:1000],outputtest[0:1000], color ='c')
plt.xlabel('Real Output')
plt.ylabel('Machine estimate')
plt.title('Prediction on Chi')
fig1.add_subplot(2,2,2)
plt.plot(outputtest[0:500,:], color = 'red', label = 'Real data')
plt.plot(clementanswer[0:500,:], color = 'blue', label = 'Predicted data from Machine')
plt.title('Prediction on Chi')
plt.legend()
plt.show()
