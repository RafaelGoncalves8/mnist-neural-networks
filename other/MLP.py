#!/usr/bin/env python
# coding: utf-8

# # MLP

# In[1]:


import numpy as np
import pandas as pd

import random
random.seed(42)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import torch
torch.random.seed = 42
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


get_ipython().system('cat ../data/wines_classification/winequality.names')


# In[3]:


get_ipython().system('head ../data/wines_classification/winequality-red.csv')


# In[4]:


df1 = pd.read_csv("../data/wines_classification/winequality-red.csv", sep=";") # skip feature names
df2 = pd.read_csv("../data/wines_classification/winequality-white.csv", sep=";")
df = pd.concat([df1, df2])

df.head()


# In[5]:


data = torch.tensor(df.sample(frac=1).values, dtype=torch.float)
data


# In[6]:


mu = np.mean(data.numpy(), 0)
std = (np.max(data.numpy(), 0) - np.min(data.numpy(), 0))

data_norm = torch.zeros_like(data)

for i, l in enumerate(data):
    for j, e in enumerate(l):
        data_norm[i, j] = (e - mu[j])/std[j]


# In[7]:


X = data_norm[:,0:11]
y = data_norm[:, 11]


# In[8]:


X_train = X[:4548, :]
y_train = y[:4548]

X_test = X[4548:, :]
y_test = y[4548:]


# In[9]:


M, N = X_train.size()
print(M, N)


# In[10]:


class Net(nn.Module):
    
    def __init__(self, N, O):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N, O)
        self.fc2 = nn.Linear(O, 1)
        self.th = nn.Tanh()
        
    def forward(self, x):
        x = self.th(self.fc1(x))
        x = self.fc2(x)
        return x
    
net = Net(N, 20)
print(net)


# In[11]:


list(net.parameters())


# In[12]:


criterion = nn.MSELoss(reduction='mean')


# In[13]:


def batch_train_step(net, X, y, alpha):
    net.zero_grad()
    output = net(X_train)
    loss = criterion(output, y_train)
    #loss = loss/N
    loss.backward()
    for f in net.parameters():
        f.data.sub_(f.grad.data * alpha)
    return loss.detach()


# In[14]:


def batch_train(net, X, y, X_test, y_test, alpha=0.1, epsilon=0.001, gamma=100, max_iter=1000):
    mse_test_vec = []
    mse_train_vec = []
    mse_test = epsilon+1
    mse_min = 9999
    counter = 0
    iter = 0
    
    while (counter < gamma and mse_test > epsilon and iter < max_iter):
        output = net(X_test)
        mse_test = criterion(output, y_test).detach()
        mse_test_vec.append(mse_test)
        mse_train = batch_train_step(net, X, y, alpha)
        mse_train_vec.append(mse_train)

        
        if (mse_test >= mse_min):
            counter += 1
        else:
            mse_min = mse_test
            counter = 0
        
        iter += 1

    return (mse_train_vec, mse_test_vec)
        


# In[15]:


epsilon = 0.01
alpha = 0.1
gamma = 50
max_iter = 1000


# In[16]:


for O in [10, 30, 80, 150]:
    for alpha in [0.003, 0.006, 0.01, 0.03, 0.06]:
        print("O:", O, "alpha:", alpha)
        net = Net(N, O)
        
        (a, b) = batch_train(net, X_train, y_train, X_test, y_test,
                            alpha, epsilon, gamma, max_iter)
        
        print("MSE_train:", np.min(a), "MSE_test:", np.min(b))
        plt.plot(a, "r")
        plt.hold = True
        plt.plot(b, "b")
        plt.hold = False
        plt.show()


# In[ ]:





# In[ ]:




