#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[1]:


#미분의 나쁜 구현의 예
def numerical_diff(f,x):
    h = 1e-50 #반올림 오차 (rounding error)문제를 일으킴
    return (f(x+h)-f(x))/h #전방차분


# In[4]:


np.float32(1e-50)


# In[9]:


#개선된 미분 구현 - 1e-4 , 중앙차분

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h) #중앙 차분


# $0.01x^2+0.1x$ 

# In[6]:


def function_1(x):
    return 0.01*x**2 +0.1*x


# In[7]:


import matplotlib.pylab as plt 

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()


# In[10]:


numerical_diff(function_1, 5)


# In[11]:


numerical_diff(function_1, 10)


# In[12]:


def tangent_line(f,x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


# In[13]:


x = np.arange(0.0,20.0,0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x,y)
plt.plot(x,y2)
plt.show()


# ##### 편미분
# $ f(x_0, x_1) = x_0^2 + x_1^2 $

# In[14]:


def function_2(x):
    return x[0]**2 + x[1]**2 #인수x : 넘파이 배열
    # return np.sum(x**2)


# In[15]:


#x0=3, x1=4 일때 x0에 대한 편미분 

def function_tmp1(x0):
    return x0*x0 + 4.0**2

numerical_diff(function_tmp1, 3.0)


# In[17]:


#x0=3, x1=4 일때 x1에 대한 편미분 

def function_tmp2(x1):
    return 3.0**2 + x1*x1

numerical_diff(function_tmp2, 4.0)

