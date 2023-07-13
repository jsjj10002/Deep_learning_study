#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a = np.array([0.3,2.9,4.0])


# In[3]:


exp_a = np.exp(a)
print(exp_a)


# In[4]:


sum_exp_a = np.sum(exp_a)
print(sum_exp_a)


# In[5]:


y = exp_a/sum_exp_a
print(y)


# In[6]:


#implement softMax function 

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    
    return y


# In[7]:


a = np.array([1010, 1000, 900])
np.exp(a)/np.sum(np.exp(a))


# In[8]:


c = np.max(a)
a-c


# In[9]:


np.exp(a-c)/np.sum(np.exp(a-c))


# In[10]:


#Softmax function improves overflow problem
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    
    return y


# In[11]:


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)


# In[12]:


np.sum(y)

