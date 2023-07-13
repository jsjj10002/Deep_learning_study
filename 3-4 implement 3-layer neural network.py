#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#inout layer
X = np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1=np.dot(X,W1)+B1


# In[3]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[4]:


Z1=sigmoid(A1)
print(A1)
print(Z1)


# In[5]:


#Hidden layer
W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)


# In[6]:


#output layer
def identity_function(x):
    return x

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
Y=identity_function(A3) #Y=A3


# ## 구현 정리

# In[7]:


import numpy as np
#implement activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x
#Initialize weight & bias
def init_network():
    network = {} #save weight and bias at dictionary
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network
#Convert input signal to output-순전파
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(Z1, W2) + b2
    Z2 = sigmoid(a2)
    a3 = np.dot(Z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)

