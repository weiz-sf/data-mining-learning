# -*- coding: utf-8 -*-

## Problem 1: Skip-Gram (50 pts = 40 + 10)

In this problem, you are goint to implement skip-gram model with negative sampling in Pytorch, apply it on the 20-newsgroup dataset, and compare your SkipGram implementation with the gensim implementation by looking at top-10 most similar words with "pittsburgh". Please note that, your SkipGram and gensim skipgram don't have to have exactly same results.

Hint:
* Running time would be long, please start early and be patient. You can reduce the number of iteration **itr_num** to 1 when you are debugging, but make sure to use **itr_num=20** to report your results.
* You may find this tutorial for the gensim library helpful if you want to get familiar with gensim: https://radimrehurek.com/gensim/auto_examples/index.html#core-tutorials-new-users-start-here

Suggestions:
* Please think about which parameters you need to define.
* Please make sure you know what shape each operation expects. Use .view() if you need to
  reshape.
  
Possible ERROR message for the code skeleton (The code skeleton is bug-free, this ERROR message is only caused by setting issues):
* You may get error message: **ValueError: unable to read local cache '/Users/emilywang/gensim-data/information.json' during fallback, connect to the Internet and retry**. Here, "Users/emilywang/" should be your own path for gesim-data.
* This indicates the gensim-data folder on your device does not include the information.json file.
* To solve this problem, you should put the provided information.json file (in our homework zip file) under the indicated path.
* For MAC users, you may see this error afterwards: **<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:777)>.** You can follow: https://timonweb.com/python/fixing-certificate_verify_failed-error-when-trying-requests-html-out-on-mac/ to solve this problem.
"""

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# load dataset

from gensim.parsing.preprocessing import preprocess_string
import gensim.downloader as api

word2id = {}
id2word = {}
sent_ids  = []
sent_wds  = []

word_count = {}

itr_num = 20

dataset = api.load("20-newsgroups")  # load dataset as iterable

# data processing

for data in dataset:
    doc = data['data']
    words = preprocess_string(doc)
    for word in words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1

MIN_COUNT = 5    # Only consider words whose frequency is larger than MIN_COUNT
WINDOW_SIZE = 2  # 2 words to the left, 2 to the right
for data in dataset:
    doc = data['data']
    words = preprocess_string(doc)
    sent_id = []
    sent_wd = []
    for word in words:
        if word_count[word] < MIN_COUNT:
            continue
        if word not in word2id:
            idx = len(id2word)
            word2id[word] = idx
            id2word[idx]  = word
        sent_id += [word2id[word]]
        sent_wd += [word]
    if len(sent_wd) <= WINDOW_SIZE * 2:
        continue
    sent_ids += [sent_id]
    sent_wds += [sent_wd]
    
data = []
for sent in sent_ids:
    for i in range(WINDOW_SIZE, len(sent) - WINDOW_SIZE):
        context = [sent[i - WINDOW_SIZE: i] + sent[i+1: i + WINDOW_SIZE + 1]]
        target  = sent[i]
        data.append((context, target))
print("data_length:",len(data))

class SkipGram(nn.Module):

    def __init__(self, vocab_size, hidden_size = 100):
        super(SkipGram, self).__init__()
        self.u_emb = nn.Embedding(vocab_size, hidden_size) #output
        self.v_emb = nn.Embedding(vocab_size, hidden_size) #input

    def forward(self, idx):
        return self.u_emb(idx)
    def loss(self, pos_data, neg_data):
        '''
            TODO: 
                Fill in this blank: Train the word embedding based on Skip-gram algorithm
        '''
#TODO   

        pos_target = []
        pos_context = []
        for context, target in pos_data:
          pos_context.append(context[0]) 
          pos_target.append([target] * len(context[0]))
        pos_context = np.array(pos_context).flatten() 
        pos_target = np.array(pos_target).flatten()

        target = Variable(torch.LongTensor(pos_target))
        context = Variable(torch.LongTensor(pos_context))
        v = self.v_emb(target)
        u = self.u_emb(context)
        self.log_sigmoid = nn.LogSigmoid()
        positive_val = self.log_sigmoid(torch.sum(u * v, dim = 1)).squeeze()
        
        neg_words = Variable(torch.LongTensor(neg_data))
        u_hat = self.u_emb(neg_words)
        neg_vals = torch.bmm(u_hat, v.unsqueeze(2)).squeeze()

        neg_val = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val
        return -loss.mean()/batch_size
#TODO

# apply the SkipGram model to the 20-newsgroup dataset

skipgram = SkipGram(len(word2id))
optimizer = optim.Adam(skipgram.parameters())

vocabulary = {key: value for key, value in word_count.items() if value >= MIN_COUNT}
N =sum(vocabulary.values())
word_prob = {key: value/N for (key, value) in vocabulary.items()}
word_ID = list(word_prob.keys())
word_ID = [word2id[word] for word in word_ID]
neg_sample_count = 10

batch_size = 1280
l = []
for i in range(itr_num):
    print("iteration: ",i)
    s = 0
    for bid in range(len(data) // batch_size):
        #print("iteration: ",i, ", batch: ", bid) # there are around 2000 batches per iteration, you may want to print the batch number to check the curret progress
        optimizer.zero_grad()
        positive_data = data[bid * batch_size : (bid + 1) * batch_size]
        '''
            TODO: 
                Conduct negative sample for negative words, based on word frequency
        '''
        #TODO
        neg_data =  np.random.choice(word_ID,size = (4*batch_size, neg_sample_count), p=list(word_prob.values()))
        #TODO 
        loss = skipgram.loss(positive_data, neg_data)
        loss.backward()
        s += loss
        optimizer.step()
    l.append(s.item()/(len(data) // batch_size))
    print("Average Loss for the current iteration: ", l[i])
    print("-----------------------------------")

# Compare the results with standard gensim implementation


from gensim.models import Word2Vec
# model = Word2Vec(sent_wds, min_count=1, window=2, size = 100, workers = 4)
model = Word2Vec(sent_wds)
print(model.wv.most_similar('pittsburgh'))

"""## Problem 2: PageRank (50 pts)

In this problem, you are going to do some proofs for the PageRank algorithm, then implement it and apply the implemented model on a citation dataset. Finally, you are going to extend it to personalized PageRank.

Please download the citation dataset from https://aminer.org/dblp_citation (Version 1). In the page, you will be able to see a very detailed README regarding the organization of the dataset.

### Part 1: PageRank Score Without Teleport (10pts)

Prove that, for a connected undirected graph, where the adjacency matrix $A = A^T$ , the PageRank score (without teleport) for node i is proportional to its degree $d_i$, i.e., $r_i = d_i/2|E|$, where |E| is the total number of edges in the graph.

#### Write Your answer here:

[Your Answer] https://drive.google.com/file/d/1fh1Q3fV3aqmcAs8mFhtg6KI_44rStOt9/view?usp=sharing

### Part 2: PageRank Score With Teleport (10pts)

Prove that, the closed form solution to PageRank with teleport is: 

$r = (1 − \beta)(I − \beta M)^{-1}  \mathbb{1}/N$


where 1 − β is the teleport probability, $M = (D^{-1}A)^T$ , $\mathbb{1}$ is the all one vector with dimentionality N, and N is the total number of nodes in the graph.

#### Write Your answer here:

[Your Answer]  https://drive.google.com/file/d/1fh1Q3fV3aqmcAs8mFhtg6KI_44rStOt9/view?usp=sharing

### Part 3: Implement PageRank With Teleport (10 pts)

Implement PageRank with teleport on Conference citation network. Show the top 50 conferences according to their PageRank scores.
"""

# import libraries

from scipy.sparse import coo_matrix
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv

from google.colab import drive

drive.mount('/gdrive')

def preprocessing():
#     parse the raw dataset, 
#     extract useful information, 
#     return the parsed entities.

    with open('/gdrive/MyDrive/Colab Notebooks/CS247/hw5/DBLPOnlyCitationOct19.txt') as file:
        id_pub, id_cite = {}, {}
        _cite_temp = []
        pub_list = {}
        for line in file:
            if not line.find('#c'):
                pub = line[2:-1]
                if pub == '':
                    pub = 'noname'
                if pub not in pub_list:
                    pub_list[pub] = {}
            if not line.find('#index'):
                paper_id = int(line[6:])
            if not line.find('#%'):
                _cite_temp.append(int(line[2:]))
            if line == "\n":
                id_pub[paper_id] = pub
                id_cite[paper_id] = _cite_temp
                _cite_temp = []
    return id_pub, id_cite, pub_list

def build_conference_citation_net(id_pub, id_cite, pub_list):
#     build conference citation network
#     return list of (conf1, conf2, weight) triples
    
    for key in id_pub:
        pub = id_pub[key]
        all_cite = id_cite[key]
        for cite in all_cite:
            _key = id_pub[cite]
            if _key in pub_list[pub]:
                pub_list[pub][_key] += 1
            else:
                pub_list[pub][_key] = 1   
    pub_encode = {}
    ind = 0
    pub_name = []
    for key in pub_list:
        pub_encode[key] = ind
        pub_name.append(key)
        ind += 1   
    
    pub_row, pub_col, value = [],[],[]

    for key1 in pub_list:
        for key2 in pub_list[key1]:
            pub_row.append(pub_encode[key1])
            pub_col.append(pub_encode[key2])
            value.append(pub_list[key1][key2])
    return zip(pub_row,pub_col,value), pub_encode, pub_name

def normalize(matrix):
#     row normalization
    rowsum = np.array(matrix.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(matrix)
    return mx

def pagerank(adj, beta):
    """
        TO DO: 
            compute pagerank scores and return them in numpy array form
    """
    #TODO
    n, _ = adj.shape
    r = np.asarray(adj.sum(axis=1)).reshape(-1)
    k = r.nonzero()[0]
    D = sp.csr_matrix((1 / r[k], (k, k)), shape=(n, n))
    #if personalize is NONE
    s = np.ones(n).reshape(n,1)/n
    I = np.eye(n)
    ranks = sp.linalg.spsolve((I - beta * adj.T @ D), s)

    ranks = ranks/ranks.sum()
    return ranks
    #TODO

# Apply your pagerank to the citation network

id_pub, id_cite, pub_list = preprocessing()
network, pub_encode, pub_name = build_conference_citation_net(id_pub, id_cite, pub_list)

col, row, value = zip(*network)
adj = coo_matrix((np.array(value), (np.array(row), np.array(col))), dtype = float, shape=(len(pub_list),len(pub_list)))

beta = 0.8
scores = pagerank(adj, beta)
ind = np.argsort(scores)
print ('top 50 conferences')
rank = 1
for i in ind[-50:]:
    print (rank, ': ', pub_name[i])
    rank = rank + 1

"""### Part 4: Personalized-PageRank (20 pts = 5 + 15)

For Personalized-PageRank, it is natural to extend queries from single node to a set of nodes. 

1. Please write down the iterative formula for computing P-PageRank when the query is a set of nodes, and explain why it is designed in the proposed way.
2. Please implement the Personalized-PageRank, and show the top-10 most similar conferences to {KDD}, {ICML}, and {KDD, ICML} on the conference citation network.

#### Write Your answer here:

[Your Answer]  https://drive.google.com/file/d/1fh1Q3fV3aqmcAs8mFhtg6KI_44rStOt9/view?usp=sharing
"""

def person_pagerank(adj, beta, target_set):
    """
    To DO:
        compute personalized-pagerank scores and return them in numpy array form
    """
    n, _ = adj.shape
    r = np.asarray(adj.sum(axis=1)).reshape(-1)
    k = r.nonzero()[0]
    D = sp.csr_matrix((1 / r[k], (k, k)), shape=(n, n))
    
    personalize = np.zeros(n)
    personalize[target_set] = 1/len(target_set)
    personalize = personalize.reshape(n, 1)

    I = np.eye(n)
    ranks = sp.linalg.spsolve((I - beta * adj.T @ D), personalize)
    ranks /= ranks.sum()
    return ranks

set_KDD = [pub_encode['KDD']]
set_ICML = [pub_encode['ICML']]
set_both = [pub_encode['KDD'],pub_encode['ICML']]

# apply the personalized-pagerank to find top-10 related conferences for KDD
beta = 0.8
scores = person_pagerank(adj, beta, set_KDD)
ind = np.argsort(scores)
print ('top 10 conferences for KDD')
rank = 1
for i in ind[-10:]:
    print (rank, ': ', pub_name[i])
    rank += 1

# apply the personalized-pagerank to find top-10 related conferences for ICML
beta = 0.8
scores = person_pagerank(adj, beta, set_ICML)
ind = np.argsort(scores)
print ('top 10 conferences for ICML')
rank = 1
for i in ind[-10:]:
    print (rank, ': ', pub_name[i])
    rank += 1

# apply the personalized-pagerank to find top-10 related conferences for KDD and ICML
beta = 0.8
scores = person_pagerank(adj, beta, set_both)
ind = np.argsort(scores)
print ('top 10 conferences for KDD and ICML')
rank = 1
for i in ind[-10:]:
    print (rank, ': ', pub_name[i])
    rank += 1
