#!/usr/bin/env python
# coding: utf-8

# In[14]:


import torch
import torchvision
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print(torch.__version__)


# ## Load the MNIST dataset

# In[15]:


mnist_trainset = datasets.MNIST(root='../data', train=True, download=True,
                                transform=None)


# In[16]:


mnist_testset = datasets.MNIST(root='../data', train=False, download=True,
                               transform=None)


# In[9]:


len(mnist_trainset), len(mnist_testset)


# ## Net architecture

# In[ ]:




