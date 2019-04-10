#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
torch.random.seed = 42

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the MNIST dataset

# In[2]:


mnist_trainset = datasets.MNIST(root='../data', train=True, download=True,
                                transform=None)

mnist_testset = datasets.MNIST(root='../data', train=False, download=True,
                               transform=None)


# In[3]:


print(mnist_trainset)
print('')
print(mnist_testset)


# In[4]:


mnist_trainset.data[0].size()


# In[5]:


size_len = mnist_trainset.data[0].size()[0]


# In[6]:


fig = plt.figure(1)
for i, img in enumerate(mnist_trainset.data[0:5]):
    ax = fig.add_subplot(1,5,i+1)
    ax.set_axis_off()
    ax = plt.imshow(img)
plt.show()


# ## Net architecture and train/test routines

# In[7]:


class Net(nn.Module):
    """MLP with 4 ReLU hidden layers and 1 softmax output layer"""
    
    def __init__(self, H, C):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(size_len*size_len, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, C)
        self.fc5 = nn.Linear(C, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        
    def forward(self, x):
        x = x.view(-1, size_len*size_len)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x


# In[8]:


def train(model, x_train, y_train, optimizer, criterion, epoch):
    model.train()
    
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    print("Train Epoch: {}\tLoss: {:.6f}".format(epoch, loss.item()))


# In[9]:


def test(model, x_test, y_test, criterion):
    model.eval()

    with torch.no_grad():
        output = model(x_test)
        test_loss = criterion(output, y_train)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss))


# ## Training

# In[10]:


X_train = mnist_trainset.data.float()
y_train = mnist_trainset.targets

X_test = mnist_testset.data.float()
y_test = mnist_testset.targets


# In[11]:


model = Net(100, 10)
print(model)


# In[12]:


alpha = 0.1
gamma = 10
max_epoch = 100
optimizer = optim.SGD(model.parameters(), lr=alpha)
criterion = torch.nn.NLLLoss()


# In[13]:


epoch = count = 0


# In[14]:


for epoch in range(100):
    train(model, X_train, y_train, optimizer, criterion, epoch)
    test(model, X_test, y_test, criterion)


# In[ ]:





# In[ ]:




