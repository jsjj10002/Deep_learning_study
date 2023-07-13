#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


A = np.array([1,2,3,4])
print(A)


# In[3]:


#Dimensions of the array
np.ndim(A)


# In[4]:


A.shape #Return a tuple


# In[5]:


A.shape[0]


# In[6]:


B = np.array([[1,2],[3,4],[5,6]])
print(B)


# In[7]:


np.ndim(B)


# In[8]:


B.shape


# In[9]:


A=np.array([[1,2],[3,4]])
A.shape


# In[10]:


B=np.array([[5,6],[7,8]])
B.shape


# In[11]:


np.dot(A,B)


# In[15]:


np.dot(B,A)


# In[13]:


A=np.array([[1,2,3],[4,5,6]])
A.shape


# In[14]:


B=np.array([[1,2],[3,4],[5,6]])
B.shape


# In[16]:


np.dot(A,B)


# In[17]:


C=np.array([[1,2],[3,4]])
C.shape


# In[18]:


np.dot(A,C)


# In[19]:


np.dot(C,A)


# In[20]:


A=np.array([[1,2],[3,4],[5,6]])
A.shape


# In[25]:


B=np.array([7,8])
B.shape


# In[26]:


np.dot(A,B)


# In[27]:


#scala product in neural network
X=np.array([1,2])
X.shape


# In[28]:


W=np.array([[1,3,5],[2,4,6]])
print(W)


# In[29]:


W.shape


# In[30]:


Y=np.dot(X,W)
print(Y)

