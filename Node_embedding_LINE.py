## implement the First-order LINE (finish contrastive loss, negative sampling and training pipleline). Get embedding of karate graph, then visualize your results.


# import necessary libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange 
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sb
import networkx as nx
import numpy as np
from scipy.linalg import sqrtm 
from numpy import linalg as LA

# load dataset and set parameters

G = nx.karate_club_graph()
edges  = np.array(list(G.edges))
degree = dict(G.degree)
true_labels = np.zeros(len(G.nodes))
for i in range(len(labels)):
    if G.nodes[i]['club']=='Officer':
        true_labels[i]=1

n_epochs = 100
neg_size = 5
batchrange = 3

class Line(nn.Module):
    def __init__(self, size, embed_dim=128):
        super(Line, self).__init__()

        self.embed_dim = embed_dim
        self.nodes_embeddings = nn.Embedding(size, embed_dim)

        # Initialization
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(-.5, .5) / embed_dim

    def loss(self, v_i, v_j, negsamples):

        u_i = self.nodes_embeddings(v_i)
        u_j = self.nodes_embeddings(v_j)
        negative = -self.nodes_embeddings(negsamples)

        s_1 = F.logsigmoid(torch.sum(torch.mul(u_i, u_j), dim=1))
        s_2 = torch.mul(u_i.view(len(u_i), 1, self.embed_dim), negative)
        s_3 = torch.sum(F.logsigmoid(torch.sum(s_2, dim=2)), dim=1)

        return -torch.mean(s_1 + s_3)

#     generating batches of data.

def makeData(samplededges, negsamplesize, degree):
    sampledNodes = set()
    nodesProb = []
    sumofDegree = 0
    for e in samplededges:
        sampledNodes.add(e[0])
        sampledNodes.add(e[1])
    sampledNodes = list(sampledNodes)
    nodesProb = [pow(degree[v],3/4) for v in sampledNodes]
    sumofDegree = sum(nodesProb)
    nodesProb[:] = [x/sumofDegree for x in nodesProb]

    for e in samplededges:
        sourcenode, targetnode = e[0], e[1]
        negnodes = []
        negsamples = 0
        while negsamples < negsamplesize:
            #remove size, size default =1 
            samplednode = np.random.choice(sampledNodes, p=list(nodesProb[:]))
            if (samplednode == sourcenode) or (samplednode == targetnode):
                continue
            else:
                negsamples += 1
                negnodes += [samplednode]
        yield [e[0], e[1]] + negnodes

# training
line = Line(len(G), embed_dim=100)
opt = optim.Adam(line.parameters())
for epoch in range(n_epochs):
    for b in trange(batchrange):
        opt.zero_grad()
        edge_idx = np.random.choice(len(edges), 10)
        samplededges = edges[edge_idx]
        
        batch = list(makeData(samplededges, neg_size, degree))
        batch = torch.LongTensor(batch)
        
        # based on the generated batch, train LINE via minimizing the loss.
        v_i = batch[:,0]
        v_j = batch[:,1]
        negsamples =  batch[:,2:]
        loss = line.loss(v_i, v_j, negsamples)
        loss.backward()
        opt.step()

# TSNE visualization, with node id on

emb  = line.nodes_embeddings.weight.data.numpy()
tsne_emb = TSNE(n_components = 2, perplexity = 5, learning_rate = 10).fit_transform(emb)

plt.scatter(tsne_emb[:,0], tsne_emb[:,1], c=true_labels)
for i in range(len(tsne_emb)):
    plt.annotate(str(i), xy=(tsne_emb[i,0], tsne_emb[i,1]))
plt.show()

# heatmap visualization, check cosine similarities between all pair of nodes

res = cosine_similarity(emb) 
sb.clustermap(res)
plt.show()

