# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:25:54 2019

@author: Dr Clement Etienam
"""

from __future__ import print_function

print(__doc__)

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


start_time = datetime.now()
x = np.genfromtxt("chi_itg.dat",skip_header=1, dtype='float')
df=x

test=df
out=test
outgp=test;
test=out;
numrows=len(test)    # rows of inout
#X=np.log(df[:,0:10])
X=(df[:,0:10])
X = StandardScaler().fit_transform(X)
y=np.reshape(test[:,-1],(600000,1))

#y2=np.zeros((600000, 1))
#for i in range (numrows):
#    if y[i]==0:
#        y2[i]=-1
##    
#    elif y[i]>0:
#        y2[i]=1
#y=y2


inputtrainclass=X[0:290000,:];
#
outputtrainclass=y[0:290000,:];
outputtest=y[290000:600001,:]
numrowstest=len(outputtest)
inputtest=X[290000:numrows,:];
X_train=inputtrainclass
y_train=outputtrainclass
X_test=inputtest
y_test=outputtest
#%outputtest=y(290000+1:end,:);
p=10
fig = plt.figure(figsize=(15,20))
def run_model(model, alg_name, plot_index):
    # build the model on training data
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    # calculate the accuracy score
    accuracy =  mean_squared_error(y_test, y_pred) 
    
    return y_pred
#    outputreq=np.zeros((numrowstest,1))
#    for i in range(numrowstest):
#        outputreq[i,:]=outputtest[i,:]-np.mean(outputtest)
#        CoDdnn1=1-(LA.norm(outputtest-y_pred)/LA.norm(outputreq))
#        CoDdnn=1 - (1-CoDdnn1)**2 ;
    
    print ('Mean squared error of',alg_name,'is', accuracy)
  

# ---- Decision Tree -----------
#print( 'doing Decision tree now')
#from sklearn import tree
#
#model = tree.DecisionTreeRegressor(criterion='mse', max_depth=5)
#run_model(model, "Decision Tree", 1)
#
## ----- Random Forest ---------------
#print( 'doing Random forest now')
#from sklearn.ensemble import RandomForestRegressor
##
#model = RandomForestRegressor(n_estimators=10)
#run_model(model, "Random Forest", 2)
#
## ----- xgboost ------------
## install xgboost
## 'pip install xgboost' or https://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/39811079#39811079
#
#print( 'doing XGB regressor now')
#from xgboost import XGBRegressor
#model = XGBRegressor()
#run_model(model, "XGBoost", 3)
#
## ------ SVM Classifier ----------------
#print( 'doing support vector now')
#from sklearn.svm import SVR
#model = SVR()
#run_model(model, "SVM Classifier", 4)
#
## -------- Nearest Neighbors ----------
#print( 'doing nearest neighbour now')
#from sklearn import neighbors
#model = neighbors.KNeighborsRegressor()
#run_model(model, "Nearest Neighbors Classifier", 5)
#
## ---------- SGD Classifier -----------------
#from sklearn.linear_model import SGDClassifier
#from sklearn.multiclass import OneVsRestClassifier
#
#model = OneVsRestClassifier(SGDClassifier())
#run_model(model, "SGD Classifier", 6)
#
## --------- Gaussian Naive Bayes ---------
#from sklearn.naive_bayes import GaussianNB
#
#model = GaussianNB()
#run_model(model, "Gaussian Naive Bayes", 7)
## ----------- Neural network - Multi-layer Perceptron  ------------
print( 'doing MLP now')
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(max_iter=1000)
y_pred=run_model(model, " MLP Neural network ", 8)
y_pred=np.reshape(y_pred,(310000,1))

fig = plt.figure()
plt.plot(outputtest, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data from Machine')
plt.title('MLP-GP Prediction on Chi')
plt.legend()
plt.show()

fig = plt.figure()
#plt.subplot(2, 3, 1)
plt.scatter(y_pred,outputtest, color ='c')
plt.xlabel('Real Output')
plt.ylabel('GP estimate')
plt.title('GP prediction using Sparse on Tauth data')
plt.show()
# --------- Logistic Regression ---------

print(' Compute L2 and R2 for the machine model')
Lerrordnn=(LA.norm(outputtest-y_pred)/LA.norm(outputtest))**0.5
L_2dnn=1-(Lerrordnn**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest[i,:]-np.mean(outputtest)


#outputreq=outputreq.T
CoDdnn1=1-(LA.norm(outputtest-y_pred)/LA.norm(outputreq))
CoDdnn=1 - (1-CoDdnn1)**2 ;
print ('R2 of fit using Machine is :', CoDdnn)
print ('L2 of fit using Machine is :', L_2dnn)
#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000).fit(inputtrainclass, outputtrainclass)
#labelDA=clf.predict(inputtest)
#accuracyLo =  accuracy_score(y_test,labelDA) * 100
#    
#print ('Accuracy of Logistic Regression is', accuracyLo)