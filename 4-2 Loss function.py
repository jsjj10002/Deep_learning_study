#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[4]:


y=np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
t=np.array([0,0,1,0,0,0,0,0,0,0])


# In[3]:


#오차제곱합 SSE
def SSE(y,t):
    return 0.5*np.sum((y-t)**2)


# In[5]:


SSE(y,t)


# In[6]:


y=np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])


# In[7]:


SSE(y,t)


# In[8]:


#교차엔트로피 CEE
def CEE(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


# In[9]:


y=np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
t=np.array([0,0,1,0,0,0,0,0,0,0])
CEE(y,t)


# In[10]:


y=np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])
CEE(y,t)


# ### 미니배치

# In[14]:


import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)


# In[16]:


#무작위 10장만 산출
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# In[17]:


#np.random.choice 사용 
np.random.choice(60000,10)


# In[18]:


#배치를 지원하는 교차엔트로피 CEE

def CEE(y,t):
    if y.ndim == 1: 
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7))/batch_size #배치 사이즈로 나눠 정규화 


# In[19]:


#원-핫 인코딩이 아닐 때 미니배치 교차엔트로피

def CEE(y,t):
    if y.ndim ==1:
        t = t,reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np,log(y[np.arange(batch_size, t)]+1e-7))/batch_size

