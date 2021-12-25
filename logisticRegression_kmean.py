# -*- coding: utf-8 -*-
### Logistic Regression 
# load dataset, split train and test 

from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing 
from numpy.linalg import inv
import numpy as np
import time

X, y = load_breast_cancer(return_X_y=True)
n_data, n_features = X.shape[0], X.shape[1]
ids  = np.arange(n_data)
np.random.seed(1)
np.random.shuffle(ids)
train_ids, test_ids = ids[: n_data // 2], ids[n_data // 2: ]

train_X, train_y = X[train_ids], y[train_ids]
test_X,  test_y  = X[test_ids],  y[test_ids]

# train and test with LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 

model = LogisticRegression()
model.fit(train_X,train_y)
pred_y = model.predict(test_X)

# print the evaluation results
print(classification_report(pred_y, test_y))

"""###Your LogisticRegression Implementation

For optimizing Logistic Regression, you should implement the for-loop version (__forloop_GA_optimizer__) of Gradient Ascent algorithm, the matrix version (__matrix_GA_optimizer__) of Gradient Ascent algorithm, and matrix form Newton-Raphson algorithm (__matrix_Newton_optimizer__). Please use 1e-2 as your learning rate.

Then, you are going to get the predicted results using your own ***My_Logistic_Regression*** function with the for-loop version optimizer, the matrix version optimizer, and the Newton-Raphson optimizer. Please use variable name **my_pred_y** for your predicted test results.
"""

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#  Three different Optimizers


def forloop_GA_optimizer(W, X, y):
#     Gradient Ascent (GA) optimizer implemented with for loop.
#     Args:
#         W - Parameters, W is the weight; Beta 
#         X - Features of training batch/instance
#         y - Label(s) of training batch/instance
#     Return:
#         W_new - Updated W

    learning_rate = 0.01
    # Initialize beta with appropriate shape
    gradient =np.zeros(n_data)
    # Perform gradient ascent
    for j in range(n_features):
        gradient[j] = 0
        for i in range(n_data):
            #Output probability value by appplying sigmoid 
            p_i_beta = y - sigmoid(np.dot(X,W))
            gradient[j] = gradient[j] + p_i_beta[j]
        # Update the weights
        # It is gradient ASCENT not descent
        W[j] = W[j] + learning_rate * gradient[j]
    return W
    
def matrix_GA_optimizer(W, X, y):
#     Gradient Ascent (GA) optimizer implemented with matrix operations.
    learning_rate = 0.01
    n_data = X.shape[0]
    # Output probability value by appplying sigmoid on itr 
    y_pred = sigmoid(np.dot(X,W))
    # Calculate the gradient values
    # This is just vectorized efficient way of implementing gradient.
    gradient = np.dot(X.T,(y - y_pred))/n_data
    W_updated = W + learning_rate * gradient

    return W_updated

def matrix_Newton_optimizer(W, X, y):
#     Newton's method optimizer implemented with matrix operations.

    alpha = 0.01
    # Define first deriv
    def nabla_f(W):
      return np.dot(X.T , y - sigmoid(np.dot(X,W)))/n_data
    # Define second deriv
    def nabla2_f(W):
      diag = np.diag(sigmoid(np.dot(X,W)) * (1-sigmoid(np.dot(X,W))))
      return -(np.dot(np.dot(X.T,diag),X) + alpha*np.identity(X.shape[1]))
    # Update Weight 
    W_updated = W - np.dot(np.linalg.inv(nabla2_f(W)),nabla_f(W))
    return W_updated

# Your own Logistic Regression Function

class My_Logistic_Regression():
#     Parameters
#     ----------
#         n_features : int, feature dimension
#         optimizer  : function, one optimizer that takes input the model weights
#                      and training data to perform one iteration of optimization.

    def __init__(self, n_features, optimizer):
        self.W = np.random.rand(n_features)
        self.optimizer = optimizer
        
    def fit(self, X, y):
#         iterate through batches or samples, then update W by one optimizer          
        n_epoch = 10000
        for epoch in range(n_epoch):
            self.W = self.optimizer(self.W, X, y)
            
            
    def predict(self, X):
        proba = sigmoid(np.dot(X,self.W))
        pred = proba
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0
        return pred

# Finally, train and test with your own My_Logistic_Regression function
# Compare the accuracy and running time of both optimization versions (for-loop and matrix).


for optimizer in [forloop_GA_optimizer, matrix_GA_optimizer, matrix_Newton_optimizer]:
    start_time = time.time()

    my_model = My_Logistic_Regression(n_features, optimizer)
    my_model.fit(train_X,train_y)
    my_pred_y = my_model.predict(test_X)
  
    end_time = time.time()
    print('Training time: %d s' % (end_time - start_time))
    print(classification_report(my_pred_y, test_y))

###Kmeans Algorithm 

# generate synthetic dataset 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial import distance 

# generate synthetic dataset
syn_X, syn_y = make_blobs(n_samples=1000, centers=4, random_state=0, cluster_std=0.85)

# Visualize the blobs as a scatter plot

plt.scatter(syn_X[:, 0], syn_X[:, 1], s=10, cmap=plt.cm.Paired)
plt.show()

# Visualize with the ground-truth cluster results

plt.scatter(syn_X[:,0], syn_X[:,1], c=syn_y, s=10, cmap=plt.cm.Paired)
plt.show()

# Identify the clusters using the sklearn K-Means algorithm.

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score
kmeans = KMeans(n_clusters=4,max_iter = 100, random_state=0)
syn_y_pred = kmeans.fit_predict(syn_X)

# Visualize with the predicted results

plt.scatter(syn_X[:,0], syn_X[:,1], c=syn_y_pred, s=10, cmap=plt.cm.Paired)
plt.show()

"""### Your Kmeans Implementation and use it on the clustering task on the synthetic dataset we generated before. Please use the variable name __my_syn_y_pred__ for your predicted results.
"""

from scipy.spatial import distance
import pandas as pd

def my_KMeans(X, n_clusters = 4, max_iter=100):

#     X: multidimensional data
#     k: number of clusters
#     max_iter: number of repetitions before clusters are established
    
#     Return: class of each data point
    
    index = np.random.choice(len(X), n_clusters, replace = False)
    cts = X[index, :]
    class_p = np.argmin(distance.cdist(X, cts, 'euclidean'), axis = 1)
    for i in range(max_iter):
        cts = np.vstack([X[class_p == j, :].mean(axis = 0) for j in range(n_clusters)])
        temp = np.argmin(distance.cdist(X, cts, 'euclidean'), axis = 1)
        if np.array_equal(class_p,temp): break
        class_p = temp
    return class_p

# Identify the clusters using your own K-Means implementation.
my_syn_y_pred = my_KMeans(syn_X, n_clusters=4, max_iter = 100)

# Visualize with the ground-truth cluster results

plt.scatter(syn_X[:,0], syn_X[:,1], c=my_syn_y_pred, s=10, cmap=plt.cm.Paired)
plt.show()
