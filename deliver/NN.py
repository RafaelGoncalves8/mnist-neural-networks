#!/usr/bin/env python
# coding: utf-8

# In[18]:


import torch
import torchvision
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the MNIST dataset

# In[20]:


mnist_trainset = datasets.MNIST(root='../data', train=True, download=True,
                                transform=None)

mnist_testset = datasets.MNIST(root='../data', train=False, download=True,
                               transform=None)


# In[21]:


print(mnist_trainset)
print('')
print(mnist_testset)


# In[53]:


fig = plt.figure(1)
N = 5
for i, img in enumerate(mnist_trainset.data[0:N]):
    ax = fig.add_subplot(1,5,i+1)
    ax.set_axis_off()
    ax = plt.imshow(img)
plt.show()


# ## Net architecture

# In[ ]:




