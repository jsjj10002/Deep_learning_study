#!/usr/bin/env python
# coding: utf-8

# In[8]:


# 신경망 학습(확률적 경사 하강법(stochastic gradient descent))의 순서 - SGD
# 1-미니배치: 훈련 데이터 중 일부 무작위로 가지고 옴 - 손실함수의 값을 줄이기 위해
# 2-기울기 산출: 각 가중치 매게변수의 기울기를 구함 - 손실 함수의 값을 작게 하는 방향 제시 
# 3- 매게변수 갱신: 매게변수를 기울기의 방향으로 갱신 
# 4- 반복: 1~3 과정 반복

#MNIST 데이터셋 사용해 학습 

import numpy as np


# In[23]:


# 2층 신경망 클래스 구현
import sys, os
sys.path.append(os.pardir)
from common.functions import * 
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화-순서대로 입력층 뉴런 수, 은닉층 뉴런 수, 출력 층의 뉴런 수 
        self.params = {} # 신경망의 메게변수를 보관하는 딕셔너리변수(인스턴스 변수)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 1번째 층의 가중치 
        self.params['b1'] = np.zeros(hidden_size)
        # 1번째 층의 편향
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 2번째 층의 가중치
        self.params['b2'] = np.zeros(output_size)
        # 2번째 층의 편향 

    def predict(self, x): # 예측 실행
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    
    #x: 입력 데이터, #t: 정답 레이블 
    def loss(self, x, t): #손실함수의 값을 구함
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t): #정확도 구함 
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum( y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t): #가중치 매게변수의 기울기를 구함 
        loss_W = lambda W: self.loss(x, t)
        
        grads = {} #기울기를 보관하는 딕셔너리 변수(numerical_gradient()메서드의 반환값)
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 1번째 층의 가중치의 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 1번째 층의 편향의 기울기
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 2번째 층의 가중치의 기울기
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        # 2번째 층의 편향의 기울기
        
        return grads
        
        


# In[25]:


#예시
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
x = np.random.rand(100,784) #더미 데이터 입력(100장 분량)
y = net.predict(x)
t = np.random.rand(100,10) # 더미 정답 레이블(100장 분량)

grads = net.numerical_gradient(x, t) # 기울기 계산 


# In[ ]:


#미니배치 학습 구현 

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet


# In[ ]:




