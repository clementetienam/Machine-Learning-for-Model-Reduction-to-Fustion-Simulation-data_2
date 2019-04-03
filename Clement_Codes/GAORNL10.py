# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 08:30:34 2019

@author: Dr Clement Etienam
"""

print(__doc__)

import numpy as np
import pylab

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp_extras.kernels import ManifoldKernel

np.random.seed(1)

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((1, 2),),
                               transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                              n_restarts_optimizer=10)

#X = np.linspace(-7.5, 7.5, 100)
#y_ = np.sin(X_) + (X_ > 0)


X = open("inputactive.out") #533051 by 28
X = np.fromiter(X,float)
#X = np.reshape(X,(1000,[]), 'F')  


y = open("outputactive.out") #533051 by 28
y = np.fromiter(y,float)
#y = np.reshape(y,(1000,1), 'F')

# Visualization of prior
pylab.figure(0, figsize=(10, 8))
X_nn = gp.kernel.k2._project_manifold(X[:, None])
pylab.subplot(3, 2, 1)
for i in range(X_nn.shape[1]):
    pylab.plot(X, X_nn[:, i], label="Manifold-dim %d" % i)
pylab.legend(loc="best")
pylab.xlim(-7.5, 7.5)
pylab.title("Prior mapping to manifold")

pylab.subplot(3, 2, 2)
y_mean, y_std = gp.predict(X[:, None], return_std=True)
pylab.plot(X, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X, y_mean - y_std, y_mean + y_std,
                   alpha=0.5, color='k')
y_samples = gp.sample_y(X[:, None], 10)
pylab.plot(X, y_samples, color='b', lw=1)
pylab.plot(X, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.legend(loc="best")
pylab.xlim(-7.5, 7.5)
pylab.ylim(-4, 3)
pylab.title("Prior samples")


# Generate data and fit GP
X = np.random.uniform(-5, 5, 40)[:, None]
y = np.sin(X[:, 0]) + (X[:, 0] > 0)
gp.fit(X, y)

# Visualization of posterior
X_nn = gp.kernel_.k2._project_manifold(X[:, None])

pylab.subplot(3, 2, 3)
for i in range(X_nn.shape[1]):
    pylab.plot(X, X_nn[:, i], label="Manifold-dim %d" % i)
pylab.xlim(-7.5, 7.5)
pylab.legend(loc="best")
pylab.title("Posterior mapping to manifold")

pylab.subplot(3, 2, 4)
y_mean, y_std = gp.predict(X[:, None], return_std=True)
pylab.plot(X, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X, y_mean - y_std, y_mean + y_std,
                   alpha=0.5, color='k')
y_samples = gp.sample_y(X[:, None], 10)
pylab.plot(X, y_samples, color='b', lw=1)
pylab.plot(X, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.plot(X, y, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(-7.5, 7.5)
pylab.ylim(-4, 3)
pylab.legend(loc="best")
pylab.title("Posterior samples")

# For comparison a stationary kernel
kernel = C(1.0, (0.01, 100)) * RBF(0.1)
gp_stationary = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                                         n_restarts_optimizer=1)
gp_stationary.fit(X, y)

pylab.subplot(3, 2, 6)
y_mean, y_std = gp_stationary.predict(X[:, None], return_std=True)
pylab.plot(X, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X, y_mean - y_std, y_mean + y_std,
                   alpha=0.5, color='k')
y_samples = gp_stationary.sample_y(X[:, None], 10)
pylab.plot(X, y_samples, color='b', lw=1)
pylab.plot(X, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.plot(X, y, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(-7.5, 7.5)
pylab.ylim(-4, 3)
pylab.legend(loc="best")
pylab.title("Stationary kernel")

pylab.tight_layout()
pylab.show()
