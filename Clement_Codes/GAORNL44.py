# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:54:02 2019

@author: Dr Clement Etienam
Plot labels
Active learning 2
"""
from __future__ import print_function
print(__doc__)

from sklearn.neural_network import MLPClassifier
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linalg as LA
X = open("inpiecewise.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(10000,2), 'F') 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
(scaler.fit(X))
X=(scaler.transform(X))
y = open("outpiecewise.out") #533051 by 28
y = np.fromiter(y,float)
y = np.reshape(y,(10000,1), 'F')  

ydami=y

#ytest = open("outtestpiecewise.out") #533051 by 28
#ytest = np.fromiter(ytest,float)
#ytest = np.reshape(ytest,(10000,1), 'F')  

ytest=y
outputtest=ytest
numrowstest=len(outputtest)
inputtrainclass=X
outputtrainclass=y
matrix=np.concatenate((X,y), axis=1)
xtest=X
inputtest=xtest
print('Do the K-means clustering with 18 clusters of [X,y] and get the labels')
kmeans =KMeans(n_clusters=18,max_iter=100).fit(matrix)
dd=kmeans.labels_
dd=dd.T
dd=np.reshape(dd,(10000,1))


import matplotlib as mpl

plt.title("Two informative features, one cluster per class", fontsize='small')

plt.scatter(X[:, 0], X[:, 1], marker='o', c=np.ravel(dd))
plt.legend()
plt.show()

RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)
X_raw=X
y_raw=dd
# Isolate our examples for our labeled dataset.
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=3)
X_train = X_raw[training_indices]
y_train = y_raw[training_indices]
# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)

from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
knn = KNeighborsClassifier(n_neighbors=3)
learner = ActiveLearner(estimator=knn, X_training=X_train, y_training=y_train)
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)


unqueried_score = learner.score(X_raw, y_raw)
# Plot our classification results.
#fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
#ax.scatter(x=x_component[is_correct], y=y_component[is_correct], c='g', marker='+',label='Correct', alpha=8/10)
#ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x',label='Incorrect', alpha=8/10)
#ax.legend(loc='lower right')
#ax.set_title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score))
#plt.show()
N_QUERIES = 800
performance_history = [unqueried_score]
# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling).
for index in range(N_QUERIES):
    query_index, query_instance = learner.query(X_pool)
# Teach our ActiveLearner model the record it has requested.
    X, y = X_pool[query_index], (y_pool[query_index])
    learner.teach(X=X, y=y)


# Remove the queried instance from the unlabeled pool.
X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)
# Calculate and report our model's accuracy.
model_accuracy = learner.score(X_raw, y_raw)
print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
# Save our model's performance for plotting.
performance_history.append(model_accuracy)


# Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0, top=1)
ax.grid(True)
ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')

predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)
## Plot our updated classification results once we've trained our learner.
#fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
#ax.scatter(x=x_component[is_correct], y=y_component[is_correct], c='g', marker='+',label='Correct', alpha=8/10)
#ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x',label='Incorrect', alpha=8/10)
#ax.set_title('Classification accuracy after {n} queries: {final_acc:.3f}'.format(n=N_QUERIES, final_acc=performance_history[-1]))
#ax.legend(loc='lower right')
#plt.show()

