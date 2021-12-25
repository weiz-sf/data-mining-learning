# -*- coding: utf-8 -*-

## prove a property of the unormalized graph Laplacian matrix. Then, you are going to implement the spectral clustering and apply it on the Karate Club dataset.

# import neccessary libraries

import networkx as nx
from sklearn.cluster import KMeans
import numpy as np
from scipy.linalg import sqrtm 
from numpy import linalg as LA

# load and preprocess the Karate Club Graph

G = nx.karate_club_graph()
nx.draw(G, with_labels=True, pos=nx.spring_layout(G))

A = nx.adj_matrix(G)
A = A.todense()

def laplacian(A):
    diag = np.diag(np.squeeze(np.array(np.sum(A,axis=1))))
    inv = np.linalg.inv(sqrtm(diag))
    symmetric_laplacian = np.eye(diag.shape[0]) - np.dot(inv,A).dot(inv)
    return symmetric_laplacian

def spectral_clustering(adjacency, n_clusters):
    k_means = KMeans(n_clusters=n_clusters, random_state=43)
    symmetric_laplacian = laplacian(A)
    eigval, eigvec = np.linalg.eig(symmetric_laplacian)
    indices = np.argsort(eigval)[1:]
    matrix = []
    for i in range(2):
        index = indices[i]
        matrix.append(eigvec[:,index])
    matrix=np.squeeze(matrix).T
    matrix_normalized = matrix/np.expand_dims(np.linalg.norm(matrix,axis=1),-1)
    km = KMeans(n_clusters=2).fit(matrix_normalized)
    return km.labels_

# use the spectral clustering method on Karate Club Graph
labels = spectral_clustering(A, 2)
print(labels) # Please keep this output in your submission

# visualize the results

r_nodes = []
b_nodes = []
for i, j in enumerate(labels):
    if j == 0:
        r_nodes += [i]
    else:
        b_nodes += [i]

pos = nx.spring_layout(G)
nx.draw(G,pos, with_labels=True,
                       nodelist=r_nodes,
                       node_color='r',
                       node_size=500,
                   alpha=0.8)
nx.draw(G,pos, with_labels=True,
                       nodelist=b_nodes,
                       node_color='b',
                       node_size=500,
                   alpha=0.8)

