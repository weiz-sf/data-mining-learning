# -*- coding: utf-8 -*-


## Sentimate Classification 

###apply the multinomial naive bayes method learned in the lecture on a real-world sentiment classification dataset. 

###Sklearn Implementation: the sentimate classification task using the multinomial naive bayes function __MultinomialNB__ implemented in the sklearn package. We've provided the data processing parts, please implememt the code for training and testing, and get the probability result using ***pred_proba*** and ***pred_log_proba***.


# load dataset, split train and test 

from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian',  'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test  = fetch_20newsgroups(subset='test',  categories=categories, shuffle=True, random_state=42)

# data processing, turn the loaded data into array

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

count_vect = CountVectorizer().fit(twenty_train['data'] + twenty_test['data']) 
X_train_feature = count_vect.transform(twenty_train['data']).toarray()
X_test_feature  = count_vect.transform(twenty_test['data']).toarray()

# train and test with MultinomialNB

from sklearn.naive_bayes import MultinomialNB
import numpy as np
''' 
    Please implement train and test using sklearn MultinomialNB.
    You are expected to get the probability result using "pred_proba" and "pred_log_proba".'''
sk_model = MultinomialNB()
sk_model = sk_model.fit(X_train_feature, twenty_train.target)

sk_proba_preds = np.argmax(sk_model.predict_proba(X_test_feature), axis=1)
sk_log_proba_preds= np.argmax(sk_model.predict_log_proba(X_test_feature), axis=1)

print(sk_proba_preds)
print(sk_log_proba_preds)

###My Multinomial Naive Bayes Function

class My_MultinomialNB():
    """
    Multinomial Naive Bayes
    ==========  
    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        

    def fit(self, X, y):
###Given feature X and label y, calculate beta and pi with a smoothing parameter alpha (laplace smoothing)
        self.class_indicator = {}
        for i, c in enumerate(np.unique(y)):
            self.class_indicator[c] = i
        self.n_class = len(self.class_indicator)
        self.n_feats = np.shape(X)[1]
        
        self.beta    = np.zeros((self.n_class, self.n_feats))
        self.pi      = np.zeros((self.n_class))

        self.idx = [[] for _ in range(self.n_class)]
        for i,c in self.class_indicator.items():
            self.idx[i] = (y == c)
            self.beta[i] = (np.sum(X[self.idx[i]],axis = 0) + self.alpha)/(np.sum(X[self.idx[i]]) + self.alpha)
            self.pi[i] = self.idx[i].shape[0]/self.n_feats
        
        self.log_beta = np.log(self.beta)
        self.log_pi   = np.log(self.pi)
          
    
    def predict_proba_without_log(self, X):
###Given a test dataset with feature X, calculate the predicted probability of each data point
        prob = np.zeros((len(X), self.n_class))
                       
        for i in range(len(X)):
          for j in range(self.n_class):
            prob[i,j] = np.prod(np.power(self.beta[j],X[i]))*self.pi[j]
        return prob
    
    
    def predict_proba_with_log(self, X):
        log_prob = self.predict_log_proba_with_log(X)
        return np.exp(log_prob - np.max(log_prob, axis=1).reshape(-1, 1))
    
    
    def predict_log_proba_with_log(self, X):
###Given a test dataset with feature X, calculate the log probability of each data point
        log_prob = np.zeros((len(X), self.n_class))

        for i in range(len(X)):
          for j in range(self.n_class):
            log_prob [i,j] = np.sum(self.log_beta[j]*X[i]) + self.log_pi[j]
        
        return log_prob

# train and test with My_MultinomialNB

my_model= My_MultinomialNB()
my_model.fit(X_train_feature,twenty_train.target)
proba = my_model.predict_proba_without_log(X_test_feature)
log_proba = my_model.predict_log_proba_with_log(X_test_feature)

my_proba_preds =np.argmax(proba, axis=1)
my_log_proba_preds =np.argmax(log_proba, axis=1)

def accuracy(y_true, y_pred):
    acc = np.equal(y_true, y_pred)
    score = sum(acc)/len(acc) # calculate the percentage of the correctness
    return score
print(my_proba_preds)
print(my_log_proba_preds)
print (accuracy(twenty_test.target, my_proba_preds))
print (accuracy(twenty_test.target, my_log_proba_preds))

###Complare sklearn MultinomialNB and your own My_MultinomialNB 

def accuracy(y_true, y_pred):
    acc = np.equal(y_true, y_pred)
    score = sum(acc)/len(acc) # calculate the percentage of the correctness
    return score

# accuracy of sklearn MultinomialMB without log
print ("accuracy of sklearn MultinomialMB without log:", accuracy(twenty_test.target, sk_proba_preds))


# accuracy of My_MultinomialMB without log
print ("accuracy of My_MultinomialMB without log:",accuracy(twenty_test.target, my_proba_preds))

# accuracy of sklearn MultinomialMB with log
print ("accuracy of sklearn MultinomialMB with log:",accuracy(twenty_test.target, sk_log_proba_preds))


# accuracy of My_MultinomialMB with log 
print ("accuracy of My_MultinomialMB with log:",accuracy(twenty_test.target, my_log_proba_preds))


###Tune alpha: choose different laplacian smoothing parameter ***alpha***, including (0, 0.001, 0.01, 0.1, 1, 10, 100, 1000), show the accuracy of your model using ***pred_log_proba***. Plot the accuracy curve with different ***alpha*** using *matplotlib* package. 
    
# for different alpha, print the accuracy of your model

accs = []
alpha_list = [0, 0.001, 0.1, 1, 10, 100, 1000, 10000]
for alpha in alpha_list:
    '''
        TO DO: Train the model with different alpha, and get corresponding accuracy
    '''
    my_model= My_MultinomialNB(alpha)
    my_model.fit(X_train_feature,twenty_train.target)
    log_proba = my_model.predict_log_proba_with_log(X_test_feature)
    my_log_proba_preds =np.argmax(log_proba, axis=1)
    accs.append(accuracy(twenty_test.target,my_log_proba_preds))
print(accs)

# Visualization: plot accuracy curve with different alpha

import matplotlib.pyplot as plt

plt.figure()
plt.plot(alpha_list,accs)
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.title("Accuracy Graph")
plt.show()

