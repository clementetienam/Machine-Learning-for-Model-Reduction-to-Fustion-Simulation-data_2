# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:17:14 2019

@author: mjkiqce3
"""

print(__doc__)
import numpy as np
from piecewise.regressor import piecewise
from datetime import datetime
import matplotlib.pyplot as plt

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


inputtrainclass=X[0:1000,:];
#
outputtrainclass=y[0:1000];
#outputtest=y[290000:600001,:]
#numrowstest=len(outputtest)
#inputtest=X[290000:numrows,:];
#X_train=inputtrainclass
#y_train=outputtrainclass
#X_test=inputtest
#y_test=outputtest
model = piecewise(inputtrainclass, outputtrainclass)

a=len(model.segments)

outputtrainclass=model.predict(inputtrainclass)