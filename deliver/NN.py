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

import random
import gc

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
        self.fc4 = nn.Linear(H, H)
        self.fc5 = nn.Linear(H, C)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = x.view(-1, size_len*size_len)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x


# In[8]:


def train(model, x_train, y_train, optimizer, criterion, epoch, disp=''):
    model.train()
    
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    acc = (torch.sum((output.detach().max(dim=1)[1] == y_train), dtype=torch.float64))/len(y_train) 

    if disp=='print':
        print("Train Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(epoch, loss.item(), acc))
    elif disp=='ret':
        return loss.item()
        


# In[9]:


def test(model, x_test, y_test, criterion, disp='ret'):
    model.eval()

    with torch.no_grad():
        output = model(x_test)
        test_loss = criterion(output, y_test)
        acc = (torch.sum((output.detach().max(dim=1)[1] == y_test), dtype=torch.float64))/len(y_test)

    if disp=='print':
        print("\nTest set: Average loss: {:.4f}\tAccuracy: {:.6f}\n".format(test_loss.item(), acc))
    elif disp=='ret':
        return test_loss, acc


# ## Training

# In[10]:


X_train = mnist_trainset.data.float()
y_train = mnist_trainset.targets

X_test = mnist_testset.data.float()
y_test = mnist_testset.targets

Y_train = np.zeros((y_train.shape[0], 10))
for i, e in enumerate(y_train):
    Y_train[i,e] = 1
    
Y_test = np.zeros((y_test.shape[0], 10))
for i, e in enumerate(y_test):
    Y_test[i,e] = 1


# ### Hyperparameters

# In[11]:


model = Net(200, 10)


# In[12]:


alpha = 0.01
gamma = 10
max_epoch = 100
optimizer = optim.SGD(model.parameters(), lr=alpha)
criterion = torch.nn.NLLLoss()


# In[13]:


epoch = count = 0


# In[14]:


for epoch in range(100):
    train(model, X_train, y_train, optimizer, criterion, epoch, 'print')
#    test(model, X_test, y_test, criterion, 'print')


# In[15]:


test(model, X_test, y_test, criterion, 'print')


# In[16]:


gc.collect()


# In[18]:


model = Net(400, 10)
optimizer = optim.Adam(model.parameters())
min_error = 999
epoch = count = 0
train_loss_vec = []
test_loss_vec = []

while (epoch < 100 and count < 30):
    train_loss = train(model, X_train, y_train, optimizer, criterion, epoch, 'ret')
    test_loss, _ = test(model, X_test, y_test, criterion, 'ret')
    train_loss_vec.append(train_loss)
    test_loss_vec.append(test_loss)
    epoch += 1
    if test_loss >= min_error:
        count += 1
    else:
        min_error = test_loss
        
test(model, X_test, y_test, criterion, 'print')


# In[19]:


plt.plot(train_loss_vec, "r")
plt.hold = True
plt.plot(test_loss_vec, "b")
plt.hold = False
plt.title("Loss progression by epochs")
plt.legend(["Train loss", "Test loss"])
plt.show()


# In[20]:


gc.collect()


# In[21]:


fig = plt.figure(1)
for i, img in enumerate(X_test[:10]):
    ax = fig.add_subplot(1,10,i+1)
    ax.set_axis_off()
    ax = plt.imshow(img)
    with torch.no_grad():
        a = model(img)
    print(mnist_testset.classes[torch.argmax(a, dim=1)])
plt.show()


# In[22]:


fig = plt.figure(1)
fig.subplots_adjust(hspace=.5)

for i, img in enumerate(X_test[:36]):
    with torch.no_grad():
        a = model(img)
        
    ax = fig.add_subplot(6,6,i+1)
    ax.set_axis_off()
    ax.set_title(mnist_testset.classes[torch.argmax(a, dim=1)])
    ax = plt.imshow(img)


plt.show()


# In[23]:


r = random.randint(0,1000)

img = X_test[r]

with torch.no_grad():
    a = model(img)
    
plt.title(mnist_testset.classes[torch.argmax(a, dim=1)])
plt.imshow(img)
plt.show()


# In[31]:


def kfold(k, N, epochs, model):
    optimizer = optim.Adam(model.parameters())
    loss_avg = 0
    acc_avg = 0
    for i in range(k):
        epoch = 0
        count = 0
        min_error = 0
        while (epoch < epochs and count < epochs/10):
            train(model, X_train[i*(int(N/k)):(i+1)*(int(N/k))], y_train[i*(int(N/k)):(i+1)*(int(N/k))], optimizer, criterion, epoch, '')
            test_loss, test_acc = test(model, X_test, y_test, criterion, 'ret')
            epoch += 1
            if test_loss >= min_error:
                count += 1
            else:
                min_error = test_loss
            print('.', end='')
        loss_avg += test_loss.detach()
        acc_avg += test_acc
        print('|', end='')
    print('>')
    return loss_avg/k, test_acc/k


# In[32]:


models = []
for _ in range(6):
    H = random.randint(100,600)
    model = Net(H, 10)
    loss, acc = kfold(5, 60000, 100, model)
    models.append((H, loss, acc))        


# In[33]:


gc.collect()


# In[34]:


models


# In[ ]:




