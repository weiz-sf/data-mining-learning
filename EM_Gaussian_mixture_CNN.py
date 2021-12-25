# -*- coding: utf-8 -*-

### Implement EM for 2-d GMM: implement EM algorithm for 3-components 2d GMM on a synthetic dataset.

# import neccessary libraries
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

# function for data generation
# inputs: 
#   num_data: number of datapoints, scaler)
#   means: mean vector for each cluster, list 2d vectors
#   covariances: covariance matrix for each cluster, list of 2d matrices
#   weights: weight for each cluster, vector, summation should be 1
# output:
#   list of 2d vectors, corresponds to the data points

def generate_data(num_data, means, covariances, weights):
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data

# generate synthetic data

num_data = 500
means = [
    [5, 0], # mean of cluster 1
    [1, 1], # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
covariances = [
    [[.5, 0.], [0, .5]], # covariance of cluster 1
    [[.92, .38], [.38, .91]], # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
weights = [1/4., 1/2., 1/4.]  # weights of each cluster

np.random.seed(0)
data = generate_data(num_data, means, covariances, weights)
samples = np.array(data)


# visualize the generated data
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], color='blue')

# define the log likelihood that we want to maximize

def log_likelihood(x, mean_em, sigma_em, weight_em):
    s = np.sum([np.log(np.sum([weight_em[k]*multivariate_normal.pdf(x_i, mean=mean_em[k,:], cov=sigma_em[k,:,:]) for k in range(num_gaussian)])) for x_i in x])
    return s

# a function for visualization

import matplotlib

def visualize(mean_em, sigma_em,n_std=2):
    plt.figure()
    ax = plt.gca()
    plt.scatter(samples[:,0],samples[:,1],color='blue')
    plt.scatter(mean_em[:,0], mean_em[:,1],color='red')
    
    for k in range(num_gaussian):
        cov = sigma_em[k]
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor='red')
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_x = mean_em[k,0]
        mean_y = mean_em[k,1]
        transf = matplotlib.transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
    plt.figure()
    plt.show()

# Initialization
# mean_em corresponds to mu in lecture slides
# sigma_em corresponds to Sigma in lecture slides
# weight_em corresponds to the w (1 dim) in lecture slides (i.e. the weight for each component)
# prob corresponds to w (2 dim) in lecture slides

num_gaussian = 3 # number of components

# randomly select 3 data point as initial mean for 3 clusters
mean_em = samples[np.random.randint(low=0, high=num_data, size=num_gaussian),:] 
# use identity matrices as initial covariance matrices
sigma_em = np.array([np.identity(2)]*num_gaussian) 
# randomly assign initial weight to each cluster
weight_em = np.zeros(num_gaussian)
a = 1
for i in range(num_gaussian-1):
    weight_em[i] = np.random.uniform(0, a)
    a -= weight_em[i]
weight_em[num_gaussian-1] = a

# create an empty matrix to store wij
prob = np.zeros((num_data, num_gaussian))

# iteratively conduct E-step and M-step
from scipy.stats import multivariate_normal
epo = 1
prev_log_likelihood = log_likelihood(samples, mean_em, sigma_em, weight_em)
log_likelihood_increment = 100

# ?multivariate_normal.pdf
# start EM
# termination criterium: more than 30 epochs and the log_likelihood increment < 0.5
while (log_likelihood_increment > 0.5 or epo <= 30):
    # E step
    #probability for all datapoint j to belong to gaussian g
    prob_bot = np.zeros((num_data, num_gaussian))
    for i, x_i in enumerate(samples):
      for k in range(num_gaussian):
          prob_bot[i,k]=weight_em[k]* multivariate_normal.pdf(x_i, mean=mean_em[k,:], cov=sigma_em[k,:,:]) 
    #normalizing the probabilities so prob sums up to 1 and weight it by mean of cluster 
    e_prob = prob_bot /(np.sum(prob_bot, axis=1).reshape(500,1))

    # M Step
    #TODO 
    # calculate the fraction of points allocated to each cluster 
    f_c = np.sum(e_prob, axis = 0)

    # calculate weights
    weight_em = f_c/num_data

    # calculate mu 
    mean_em = np.dot(e_prob.T, samples)/(f_c.reshape(num_gaussian,1))
    
    # calculate the sigma
    def var_function (prob, num_samples, m_c, mean_em):
        sigma = np.zeros((3, 2, 2))
        for i in range(3):
            var_c = (1/f_c[i])*(np.dot((e_prob[:,i].reshape(num_data,1) * (num_samples - mean_em[i])).T,(num_samples - mean_em[i])))
            sigma[i] = var_c
        return sigma
    sigma_em = var_function(e_prob, samples, f_c, mean_em)
   
    # Check termination criterium
    curr_log_likelihood = log_likelihood(samples, mean_em, sigma_em, weight_em)
    log_likelihood_increment = curr_log_likelihood - prev_log_likelihood
    print("Run", epo, 'done with log likelihood:', curr_log_likelihood)
    if epo%5==1:
        visualize(mean_em, sigma_em) # visualize for every 5 epochs, you would be able to see how the centers are moving
    epo += 1
    prev_log_likelihood = curr_log_likelihood

##Neural Networks: implement a simple 2-layer NN classifier (with linear layers only), and appy it to synthetic dataset. Then, you are going to implement a 2-layer CNN classifier, and apply it to the CIFAR-10 dataset. You will have to use **pytorch**, which is a library for Python programs that facilitates building deep learning projects.

###This code has been tested on pytorch 1.8.1, it may work with other versions, but we won’t be officially supporting them.

### Toy example, which is a simple **2-layer NN classifier** (with linear layers only), and appy it to synthetic dataset.

#GPU option 
import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# import neccessary libraries

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# define hyper-parameters

toy_learning_rate = 0.02
toy_epoch_num = 29

# generate synthetic dataset and visualize it

torch.manual_seed(10) # fix random seed for reproducibility

n_data = torch.ones(200, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(200, 2)
y0 = torch.zeros(200)               # class0 y data (tensor), shape=(200, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(200, 2)
y1 = torch.ones(200)                # class1 y data (tensor), shape=(200, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (400, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (400,) LongTensor = 64-bit integer

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

# A two layer neural network

class toy_2_layer_NN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(toy_2_layer_NN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        '''
            TODO: Implement Forward Pass of this model
        '''
        # x = x.view(x.size(0), -1)
        x=self.hidden(x)
        x=self.out(x)
        return x

# define the network, optimizer, and loss function

toy_net = toy_2_layer_NN(n_feature=2, n_hidden=10, n_output=2)     # define the network
toy_optimizer = torch.optim.SGD(toy_net.parameters(), lr=toy_learning_rate) # define the optimizer
toy_loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

print(toy_net) # print model architecture

# training
# you would be able to see how the prediction results and the training accuracy get updated
# in this toy example, we aim to help you get familiar with how a neural network is implemented and trained
# therefore, we do not include a test stage in this example
# in next part, you would be able to see the full process including training and testing

plt.ion()   # something about plotting

for epoch in range(toy_epoch_num):
    
    toy_optimizer.zero_grad()   # clear gradients for next train
    out = toy_net(x)                 # input x and predict based on x
    toy_loss = toy_loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
    toy_loss.backward()         # backpropagation, compute gradients
    toy_optimizer.step()        # apply gradients

    if epoch % 4 == 1:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

"""### CNN Classifier: implement a **2-layer CNN classifier**, and apply it to the CIFAR-10 dataset.

You may find the following link helpful if you want to get familiar with the CIFAR-10 dataset:
https://www.cs.toronto.edu/~kriz/cifar.html
"""

# import neccessary libraries

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
torch.manual_seed(10) # fix random seed for reproducibility

# load dataset, split into batches

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# visualize some examples in the dataset


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# A 2-layer CNN
import torch.nn.functional as F
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''
            TODO: 
            Implement Forward Pass of this model
            Please use max pooling over a (2,2) window
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.F.relu(fc1(x))
        x = self.F.relu(fc2(x))
        x = self.fc3(x)
        return x

# define the network, optimizer, and loss function

net=CNN_Net()
#net.device 

#Define loss function and Optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(), lr=learning_rate)

# Train the network
# loop over the data iterator, and feed the inputs to the network and optimize.

loss_record = []

for epoch in range(epoch_num):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # clear gradients for next train
        optimizer.zero_grad()
        
        #TODO
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        #TODO

        
        # print statistics
        running_loss += loss.item()
        print_mini_batch_num = int(len(trainloader)/ 10)
        if i % print_mini_batch_num == print_mini_batch_num-1:    # print every print_mini_batch_num mini-batches
            print('[epoch: %d, batch: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_mini_batch_num))
            loss_record.append(running_loss)
            running_loss = 0.0

print('Finished Training')

# define hyper-parameters

learning_rate = 0.0001
epoch_num = 20
batch_size = 8

# plot the loss curve

loss_record = np.array(loss_record)
plt.plot(loss_record)

# test

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

