# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 20:49:38 2019

@author: Dr Clement Etienam
"""

from __future__ import print_function
print(__doc__)

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import ConstantKernel as C, Matern
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from gp_extras.kernels import LocalLengthScalesKernel

np.random.seed(42)

n_samples = 50

# Generate data
def f(X):  # target function
    return np.sin(5*X) + np.sign(X)

#X = np.random.uniform(-1, 1, (n_samples, 1))  # data
#y = f(X)[:, 0]



x = np.genfromtxt("chi_itg.dat",skip_header=1, dtype='float')
df=x

test=df
out=test
outgp=test;
test=out;
numrows=len(test)    # rows of inout
X1=np.log(df[:,0:10])
y1=test[:,-1]
outputtest=y1[290000:600001]
numrowstest=len(outputtest)
X=X1[0:3000,:]
y=y1[0:3000]
y=np.reshape(y,(3000,1))
# Define custom optimizer for hyperparameter-tuning of non-stationary kernel
def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, maxiter=20, disp=False, polish=False)
    return res.x, obj_func(res.x, eval_gradient=False)

# Specify stationary and non-stationary kernel
kernel_matern = C(1.0, (1e-10, 1000)) \
    * Matern(length_scale_bounds=(1e-1, 1e3), nu=1.5)
gp_matern = GaussianProcessRegressor(kernel=kernel_matern)

kernel_lls = C(1.0, (1e-10, 1000)) \
  * LocalLengthScalesKernel.construct(X, l_L=0.1, l_U=2.0, l_samples=5)
gp_lls = GaussianProcessRegressor(kernel=kernel_lls, optimizer=de_optimizer)

# Fit GPs
gp_matern.fit(X, y)
gp_lls.fit(X, y)

#print "Learned kernel Matern: %s" % gp_matern.kernel_
#print "Log-marginal-likelihood Matern: %s" \
#    % gp_matern.log_marginal_likelihood(gp_matern.kernel_.theta)
#
#
#print "Learned kernel LLS: %s" % gp_lls.kernel_
#print "Log-marginal-likelihood LLS: %s" \
#    % gp_lls.log_marginal_likelihood(gp_lls.kernel_.theta)

# Compute GP mean and standard deviation on test data
X_ = X1[3000:6000,:]
outputtest = np.reshape(y1[3000:6000,:],(3000,1))

y_mean_lls, y_std_lls = gp_lls.predict(X_[:, np.newaxis], return_std=True)
y_mean_matern, y_std_matern = \
    gp_matern.predict(X_[:, np.newaxis], return_std=True)



fig = plt.figure()
plt.plot(outputtest, color = 'red', label = 'Real data')
plt.plot(y_mean_lls, color = 'blue', label = 'Predicted data from Machine')
plt.title('LogisticRegression-GP Prediction on Chi')
plt.legend()
plt.show()


fig = plt.figure()
plt.plot(outputtest, color = 'red', label = 'Real data')
plt.plot(y_mean_matern, color = 'blue', label = 'Predicted data from Machine')
plt.title('LogisticRegression-GP Prediction on Chi')
plt.legend()
plt.show()