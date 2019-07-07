#!/usr/bin/env python
# coding: utf-8

# In[66]:


import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.random.seed = 42

import random

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
n_epochs = 5
log_interval = 10


# In[44]:


mnist_trainset = datasets.MNIST(root='../data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(32),
                                    torchvision.transforms.ToTensor(), 
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                ])
                                )

mnist_testset = datasets.MNIST(root='../data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(32),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                                    ])
                              )


# In[45]:


train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size_train, True)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size_test, True)


# In[46]:


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# In[47]:


fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig


# In[69]:


class Net(nn.Module):
    def __init__(self, H):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(D)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 6 @ 12x12
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 16 @ 3x3
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x)) # 120
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) # 84
        x = self.dropout(x)
        x = self.fc3(x)         # 10
        return F.log_softmax(x)


# In[49]:


network = Net(0)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)


# In[50]:


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


# In[51]:


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'model.pth')
      torch.save(optimizer.state_dict(), 'optimizer.pth')


# In[52]:


def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


# In[53]:


test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()


# In[54]:


fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue', zorder=1)
plt.scatter(test_counter, test_losses, color='red', zorder=2)
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()


# In[55]:


with torch.no_grad():
  output = network(example_data)


# In[56]:


fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
fig


# In[61]:


img = enumerate(test_loader)
im_idx, (im_data, im_targets) = next(img)

with torch.no_grad():
  output = network(im_data)
    
plt.imshow(im_data[0][0], cmap='gray', interpolation='none')
plt.title("Prediction: {}/{}".format(
output.data.max(1, keepdim=True)[1][0].item(),
im_targets[0]))
plt.xticks([])
plt.yticks([])

plt.show()


# In[70]:


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


# In[71]:


models = []
for _ in range(5):
    D = random.randint(0,1)
    model = Net(D)
    loss, acc = kfold(5, 60000, 100, model)
    models.append((H, loss, acc))      


# ### Referencias
# 
# https://nextjournal.com/gkoehler/pytorch-mnist

# In[ ]:





# In[ ]:





# In[ ]:




