#!/usr/bin/env python
# coding: utf-8

# # PyTorch

# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


# ## Basic

# In[3]:


x = torch.empty(5, 3)
x = torch.rand(5, 3)
x = torch.zeros(5, 4, dtype=torch.long)
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)
x = torch.randn_like(x, dtype=torch.float)

print(x)
print(x.size())

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add(x)
print(y)

print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # -1 infered from other dimensions
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# ## Autograd

# ### Tensor

# In[5]:


x = torch.ones(2, 2, requires_grad=True)
print(x)


# In[6]:


y = x + 2
print(y)


# In[7]:


print(y.grad_fn)


# In[8]:


z = y * y * 3
out = z.mean()
print(z, out)


# In[9]:


a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


# ### Gradients

# In[10]:


out.backward()


# In[11]:


print(x.grad) # grad out in respect with x (d_out/d_xi)


# In[13]:


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)


# In[14]:


v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad) # gradient of y in respect to x ? why v ?


# In[15]:


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


# ## Neural Networks

# ### Model

# In[21]:


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 image channel
        # 6 output channels, 5x5 sqyare convolution channel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # first is batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print(net)


# In[22]:


params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight


# In[23]:


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


# In[25]:


net.zero_grad()
out.backward(torch.randn(1,10))


# ### Loss function

# In[26]:


output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


# In[27]:


print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])


# ### Backprop

# In[28]:


net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# ### Update the weights

# In[29]:


learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# In[32]:


optimizer = optim.SGD(net.parameters(), lr=0.01)

# training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()


# ## Training a Classifier

# In[37]:


transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                       train=True,
                                       download=True,
                                       transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                     shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                      train=False,
                                      download=True,
                                      transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[40]:


def imshow(img):
    img = img/ 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[ ]:





# In[ ]:




