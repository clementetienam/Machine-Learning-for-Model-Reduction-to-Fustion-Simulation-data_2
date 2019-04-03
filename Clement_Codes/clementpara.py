# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:19:55 2019

@author: mjkiqce3
"""
import numpy as np
import multiprocessing
from multiprocessing import Pool
def computeerr(n,regressor,cc,inputtest):

    for i in range(n): # We can parallelise here
        datause=cc[:,:,i]
        X=datause[:,0:10]
        y=datause[:,-1]
        y=np.reshape(y,(600,1))
        print ('Now in data set :', i)
        n_queries = 10
        for idx in range(n_queries): # We cant parallelise here
            query_idx, query_instance = regressor.query(X)
            regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))
    y_pred_final= regressor.predict(inputtest)
    return y_pred_final,regressor
    
if __name__ == "__main__":
    number_of_realisations = range(100)
    p = Pool(multiprocessing.cpu_count())
    p.map(computeerr,number_of_realisations)