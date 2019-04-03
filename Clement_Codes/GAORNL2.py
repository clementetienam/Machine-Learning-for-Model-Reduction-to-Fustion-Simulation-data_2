# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:41:43 2019

@author: mjkiqce3
"""
from __future__ import print_function
from sklearn.linear_model import LogisticRegression
import numpy as np
from oct2py import Oct2Py
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

start_time = datetime.now()
x = np.genfromtxt("chi_itg.dat",skip_header=1, dtype='float')
df=x

test=df
out=test
outgp=test;
test=out;
numrows=len(test)    # rows of inout
X=np.log(df[:,0:10])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
(scaler.fit(X))
X=(scaler.transform(X))
y=test[:,-1]


y2=np.zeros((600000, 1))
for i in range (numrows):
    if y[i]==0:
        y2[i]=-1
#    
    elif y[i]>0:
        y2[i]=1
y=y2


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
    accuracy =  accuracy_score(y_test, y_pred) * 100

    
    print ('Accuracy of',alg_name,'is', accuracy)
  

# ---- Decision Tree -----------
from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
run_model(model, "Decision Tree", 1)

# ----- Random Forest ---------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=500)
run_model(model, "Random Forest", 2)

# ----- xgboost ------------
# install xgboost
# 'pip install xgboost' or https://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/39811079#39811079

from xgboost import XGBClassifier
model = XGBClassifier()
run_model(model, "XGBoost", 3)

# ------ SVM Classifier ----------------
from sklearn.svm import SVC
model = SVC()
run_model(model, "SVM Classifier", 4)

# -------- Nearest Neighbors ----------
from sklearn import neighbors
model = neighbors.KNeighborsClassifier()
run_model(model, "Nearest Neighbors Classifier", 5)

# ---------- SGD Classifier -----------------
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

model = OneVsRestClassifier(SGDClassifier())
run_model(model, "SGD Classifier", 6)

# --------- Gaussian Naive Bayes ---------
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
run_model(model, "Gaussian Naive Bayes", 7)
# ----------- Neural network - Multi-layer Perceptron  ------------
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver= 'lbfgs',max_iter=3000)
run_model(model, " MLP Neural network ", 8)
# --------- Logistic Regression ---------

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000).fit(inputtrainclass, outputtrainclass)
labelDA=clf.predict(inputtest)
accuracyLo =  accuracy_score(y_test,labelDA) * 100
    
print ('Accuracy of Logistic Regression is', accuracyLo)