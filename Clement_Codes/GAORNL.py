# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:41:43 2019

@author: mjkiqce3
"""
from __future__ import print_function

from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
from oct2py import Oct2Py

import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linalg as LA

start_time = datetime.now()
x = np.genfromtxt("chi_itg.dat",skip_header=1, dtype='float')
df=x

test=df
out=test
outgp=test;
test=out;
numrows=len(test)    # rows of inout
X=np.log(df[:,0:10])
y=test[:,-1]
outputtest=y[290000:600001]
numrowstest=len(outputtest)
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
inputtest=X[290000:numrows,:];
#%outputtest=y(290000+1:end,:);
p=10;
outGP=x[0:290000,:]
outGP = outGP[~np.any(outGP == 0, axis=1)]
outputtrainGP=np.log(outGP[:,10])
outputtrainGP = np.reshape(outputtrainGP, (107861, 1))
inputtrainGP=np.log(outGP[:,0:10])

oc = Oct2Py()
#x = oc.zeros(3,3)
#inputrainGP,outputtrainGP,inputtrainclass,outputtrainclass,inputtest=oc.DatasetRegressClass()
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000).fit(inputtrainclass, outputtrainclass)
labelDA=clf.predict(inputtest)
ff=clf.score(inputtrainclass, outputtrainclass)
clement=oc.predictclassregress(labelDA,inputtest,inputtrainGP,outputtrainGP)
print(' Compute L2 and R2 for the sparse GP model')

outputtest = np.reshape(outputtest, (numrowstest, 1))
Lerrorsparse=(LA.norm(outputtest-clement)/LA.norm(outputtest))**0.5
L_2sparse=1-(Lerrorsparse**2)
#Coefficient of determination
outputreq=np.zeros((numrowstest,1))
for i in range(numrowstest):
    outputreq[i,:]=outputtest[i,:]-np.mean(outputtest)


#outputreq=outputreq.T
CoDspa=1-(LA.norm(outputtest-clement)/LA.norm(outputreq))
CoDsparse=1 - (1-CoDspa)**2 ;
print ('R2 of fit using the machine is :', CoDsparse)
print ('L2 of fit using the machine is :', L_2sparse)