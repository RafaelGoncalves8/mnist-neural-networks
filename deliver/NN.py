#!/usr/bin/env python
# coding: utf-8

# In[62]:


import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.random.seed = 42

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the MNIST dataset

# In[63]:


mnist_trainset = datasets.MNIST(root='../data', train=True, download=True,
                                transform=None)

mnist_testset = datasets.MNIST(root='../data', train=False, download=True,
                               transform=None)


# In[64]:


print(mnist_trainset)
print('')
print(mnist_testset)


# In[74]:


M = len(mnist_trainset.data[0])
M


# In[65]:


fig = plt.figure(1)
N = 5
for i, img in enumerate(mnist_trainset.data[0:N]):
    ax = fig.add_subplot(1,5,i+1)
    ax.set_axis_off()
    ax = plt.imshow(img)
plt.show()


# ## Net architecture

# In[66]:


class Net(nn.Module):
    
    def __init__(self, M, H, C):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, C)
        self.fc5 = nn.Linear(C, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x


# In[76]:


model = Net(M, 10, 10)
print(model)


# In[78]:


alpha = 0.1


# In[80]:


optimizer = optim.SGD(model.parameters(), lr=alpha)

criterion = torch.nn.CrossEntropyLoss()


# In[81]:


x = mnist_trainset.data


# In[82]:


optimizer.zero_grad()
output = model(x)
loss = criterion(output, target)
loss.backward()
optimizer.step()


# In[ ]:




