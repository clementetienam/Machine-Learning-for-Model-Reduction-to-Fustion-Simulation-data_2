# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:43:28 2019

@author: mjkiqce3
"""

import pickle

intArray = [i for i in range(1,100)]
output = open('data.pkl', 'wb')
pickle.dump(intArray, output)
output.close()