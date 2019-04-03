
# -*- coding: utf-8 -*-
"""
Created on Tuesday Feb 05 12:05:47 2019

@author: Dr Clement Etienam
This is the code for learning a machine for discountinous Toy function
We will cluster th data first, use that label from the cluster and learn a
classifier then a regressor
This code is very important
"""

from __future__ import print_function
print(__doc__)

#from IPython import get_ipython
#get_ipython().magic('reset -sf')

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


#for i in range (numrows):
#    if y[i]==0:
#        y2[i]=-1
##    
#    elif y[i]>0:
#        y2[i]=1
print('cluster with X and y')
X = open("inpiecewise2.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(10000,2), 'F') 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
(scaler.fit(X))
X=(scaler.transform(X))
y = open("outpiecewise2.out") #533051 by 28
y = np.fromiter(y,float)
y = np.reshape(y,(10000,1), 'F')  

ydami=y

#ytest = open("outtestpiecewise.out") #533051 by 28
#ytest = np.fromiter(ytest,float)
#ytest = np.reshape(ytest,(10000,1), 'F') 
print('split for regression prblem') 
from sklearn.model_selection import train_test_split
#X_traind, inputtest, y_traind, outputtest = train_test_split(X, y, test_size=0.5)
#ytest=y
outputtest=y
X_traind=X
y_traind=y
inputtest=X
numrowstest=len(outputtest)

XX, YY = np.meshgrid(np.arange(100),np.arange(100))

fig1 = plt.figure(figsize =(8,8))

JM1=np.reshape(outputtest,(100,100),'F')



fig1.add_subplot(2,2,2)
plt.pcolormesh(XX.T,YY.T,JM1,cmap = 'jet')
plt.title('True', fontsize = 15)
plt.ylabel('2',fontsize = 11)
plt.xlabel('1',fontsize = 11)
plt.axis([0,(100),0,(100)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar15 = plt.colorbar()
cbar15.ax.set_ylabel('2D function',fontsize = 11)
plt.clim(0,500)