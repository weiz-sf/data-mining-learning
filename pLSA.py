# -*- coding: utf-8 -*-

## probabilistic Latent Semantic Analysis 

from google.colab import drive

drive.mount('/gdrive')

# import neccessary libraries

import numpy as np # please update your numpy version if you get the error "module 'numpy.random' has no attribute 'default_rng'"
from gensim.parsing.preprocessing import preprocess_string
from collections import defaultdict

# load and pre-process data
# word2id : a map mapping terms to their corresponding ids
# id2word : a map mapping ids to terms
# X : document-word matrix, N*M, each line is the number of terms that show up in the document


wordCounts = []
word2id = {}
id2word = {}

datasetFilePath = '/gdrive/MyDrive/dataset2.txt' 
# you can use dataset1 to check your implementation, but you should use dataset2 to report your results#
num_topics = 10
# for dataset1, please use num_topics=4; for dataset2, please use num_topics=10#

fin = open(datasetFilePath,encoding='UTF-8')
documents = fin.readlines()
for doc in documents:
    words = preprocess_string(doc)
    word_count = defaultdict(lambda: 0)
    for word in words:
        if word not in word2id:
            idx = len(id2word)
            word2id[word] = idx
            id2word[idx]  = word
        word_count[word] += 1
    wordCounts += [word_count]
    
    
num_doc = len(documents)
num_words = len(word2id)  

# generate the document-word matrix
X = np.zeros([num_doc, num_words], np.int)

for word in word2id:
    j = word2id[word]
    for i in range(0, num_doc):
        if word in wordCounts[i]:
            X[i, j] = wordCounts[i][word]
            

print("Number of Documents: ", num_doc)
print("Number of Words in the Vocabulary: ", num_words)

# pLSA class

class PLSA():
    def __init__(self, num_doc, num_words, num_topics):
        
        self.num_doc = num_doc
        self.num_words = num_words
        self.num_topics = num_topics
        
        # theta_dz: topic distribution for each document p(z|d)
        np.random.seed(0)
        self.theta = np.random.random([num_doc, num_topics])

        # beta_zw:  word distribution for each topic p(w|z) 
        np.random.seed(1)
        self.beta = np.random.random([num_topics, num_words])

        # p[i, j, k] : lower bound of p(zk|di,wj)
        self.p = np.zeros([num_doc, num_words, num_topics])

        self.theta /= np.sum(self.theta, axis=1).reshape(-1,1)
        self.beta  /= np.sum(self.beta, axis=1).reshape(-1,1)

    def EStep(self):
        for i in range(0, num_doc):
          for j in range(0, num_words):
              denominator = 0;
              for k in range(0, num_topics):
                  self.p[i, j, k] = self.beta[k, j] * self.theta[i, k];
                  denominator += self.p[i, j, k];
              if denominator == 0:
                  for k in range(0, num_topics):
                      self.p[i, j, k] = 0;
              else:
                  for k in range(0, num_topics):
                      self.p[i, j, k] /= (denominator + 0.0000000001) 
                        
    def MStep(self, X):
        for k in range(0, num_topics):
          denominator = 0
          for j in range(0, num_words):
              self.beta[k, j] = 0
              for i in range(0, num_doc):
                  self.beta[k, j] += X[i, j] * self.p[i, j, k]
              denominator += self.beta[k, j]
          if denominator == 0:
              for j in range(0, num_words):
                  self.beta[k, j] = 1.0 / num_words
          else:
              for j in range(0, num_words):
                  self.beta[k, j] /= (denominator + 0.0000000001)   
        
        for i in range(0, num_doc):
            for k in range(0, num_topics):
                self.theta[i, k] = 0
                denominator = 0
                for j in range(0, num_words):
                    self.theta[i, k] += X[i, j] * self.p[i, j, k]
                    denominator += X[i, j];
                if denominator == 0:
                    self.theta[i, k] = 1.0 / num_topics
                else:
                    self.theta[i, k] /= (denominator +0.0000000001 )
                    
    # calculate the log likelihood
    def LogLikelihood(self, X):
        loglikelihood = 0
        for i in range(0, self.num_doc):
            for j in range(0, self.num_words):
                tmp = 0
                for k in range(0, self.num_topics):
                    tmp += self.beta[k, j] * self.theta[i, k]
                if tmp > 0:
                    loglikelihood += X[i, j] * np.log(tmp)
        return loglikelihood

# test the PLSA class

plsa = PLSA(num_doc, num_words, num_topics)

# EM algorithm
for i in range(0, 30):
    plsa.EStep()
    plsa.MStep(X)
    log_likelihood = plsa.LogLikelihood(X)
    print("iteration %d, LogLikelihood: %.3f" % (i, log_likelihood))

# print the top-5 frequent words in each topic

for i in plsa.beta:
    topic_word = []
    for idx in (-i).argsort()[:5]:
        topic_word += [id2word[idx]]
    print(topic_word)
