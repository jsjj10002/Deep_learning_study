#!/usr/bin/env python
# coding: utf-8

# In[6]:


#부모 디렉터리의 파일을 가져올 수 있도록 설정
import sys , os
sys.path.append(os.pardir)
#.dataset/mnist.py의 load_mnist를 임포트 
from dataset.mnist import load_mnist


# In[7]:


#(훈련이미지,훈련레이블), (시험 이미지, 시험 레이블) 형태로 반환
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#인수로 nomalize, flatten, one_hot_label 설정가능 (bool)


# In[8]:


print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


# ### MNIST 데이터를 화면에 불러오기

# In[11]:


import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) #넘파이로 변환된 데이터 PIL용 데이터 객체로 변환 
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28) #flatten=True로 설정해 1차원 넘파이로 저장된 데이터 28*28 크기로 변환 
print(img.shape)

img_show(img)


# ### 신경망 구축

# In[16]:


import pickle


# In[19]:


#활성함수 
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    return np.exp(a-c)/np.sum(np.exp(a-c))


# In[23]:


def get_data():
    (x_train, t_train), (x_test, t_test) =     load_mnist(flatten=True, normalize=True, one_hot_label=False)
    #nomalize=True 각 픽셀 값을 0.0~1.0으로 정규화(전처리) 
    #flatten=True 2차원을 1차월으로 가지고옴 28*28->784
    return x_test, t_test

#pickle파일에 저장되어 있는 '학습된 가중치 매개변수'를 읽음-가중치 편향 매개변수 딕셔너리로 저장
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = softmax(a3)
    
    return y
    


# In[26]:


#정확도 계산 
x, t = get_data() #mnist 데이터 얻음 
network = init_network() #네트워크 생성

accuracy_cnt = 0
for i in range(len(x)): #데이터 하나씩 분류 - 확률을 넘파이 배열로 분류 
    y = predict(network, x[i]) 
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스 얻음 - 예측 결과 
    if p == t[i]: #답과 비교해 맞춘 갯수 측정 
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt)/len(x)) ) #정확도 구함 


# In[27]:


x,_ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']


# In[28]:


x.shape


# In[29]:


x[0].shape


# In[30]:


W1.shape


# In[31]:


W2.shape


# In[32]:


W3.shape


# ### 배치처리 구현

# In[34]:


x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size] #사진을 100장씩 묶어서 꺼냄 
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) #1번째 차원을 구성하는 각 원소에서 최댓값 인덱스를 찾음 - 0번째, 1번째 
    accuracy_cnt += np.sum(p==t[i:i+batch_size]) # True가 몇 개인지 셈
    
print("Accuracy:" + str(float(accuracy_cnt)/len(x)) )


# In[39]:


#axis예시
x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])
y = np.argmax(x, axis=1)
print(x)
print(y)

