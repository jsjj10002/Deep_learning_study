
import numpy as np
#inout layer
X = np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])
print(W1.shape) # (2,3)
print(X.shape) # (2, )
print(B1.shape) # (3, )
#입력 층에서 1층으로 
A1=np.dot(X,W1)+B1
#활성화 함수-시그모이드 
def sigmoid(x):
    return 1/(1+np.exp(-x))

Z1=sigmoid(A1) #가중치 합을 활성홤수로 변환시킴 
print(A1)
print(Z1) #넘파이 배열을 반환 

#Hidden layer
W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)

#output layer
#항등 함수 정의 
def identity_function(x):
    return x # 입력을 그대로 출력 

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
Y=identity_function(A3) #Y=A3

#출력층의 활성화 함수: 풀고자 하는 문제의 성질에 맞게 정함 

# ## 구현 정리

import numpy as np
#implement activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x
#Initialize weight & bias
def init_network():
    network = {} #save weight and bias at dictionary #각 층에 필요한 매게변수 저장 
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

network = init_network() #가중치와 편향을 초기화, 이들을 딕셔너리 변수인 network에 저장 
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)

