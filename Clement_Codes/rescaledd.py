# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 10:07:07 2018

@author: mjkiqce3

"""

def rescc():
    
   
    import numpy as np
	 #x = np.reshape(x,(-1,10))   
    from sklearn.preprocessing import MinMaxScaler

    print('Standardize and normalize the input data')
	mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)
	x2= np.reshape(s,(10,10), 'F')

  
    scaler = MinMaxScaler(feature_range=(0, 1))
    input1 = scaler.fit_transform(B)
    return input1
