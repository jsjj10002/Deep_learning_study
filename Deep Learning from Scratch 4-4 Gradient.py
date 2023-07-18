#그래디언트 : 모든 변수의 편미분을 벡터로 정리한 것 
import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2 #인수x : 넘파이 배열
    # return np.sum(x**2)

#기울기(모든 변수의 편미분을 백터로 정리) 구현
def numerical_gradient(f,x): #인수 f는 함수, x는 넘파이 배열
    h=1e-4
    grad=np.zeros_like(x) # x와 형상이 같고 그 원소가 모두 0인 배열을 만듦
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h #f(x+h)
        fxh1 = f(x)
        
        x[idx] = tmp_val - h #f(x-h)
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_val
        
    return grad

#세 점 (3,4), (0,2), (3,0)에서의 기울기 구하기 
numerical_gradient(function_2, np.array([3.0, 4.0]))
# array([6., 8.])
numerical_gradient(function_2, np.array([0.0, 2.0]))
# array([0., 4.])
numerical_gradient(function_2, np.array([3.0, 0.0]))
# array([6., 0.])
# 기욹가 가르키는 쪽: 각 장소에서 함수의 출력값을 가장 크게 줄이는 방향 

# 경사 하강법 (gradient descent methood): 손실함수가 최솟값일 때의 매게변수 값 찾기 
#손실함수의 기울기가 가리키는 곳 (기울기가 0인 곳 )-반드시 최소는 아님 

#경사하강법 구현
# 학습률: 한 번의 학습으로 얼마나 학습하는지  
def gradient_descent(f, init_x, lr=0.01, step_num=100): #lr: 학습률, step_num: 반복횟수 , f: 최적화 하려는 함수 
    x = init_x #init_x: 초기값 
    
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    return x


# 경사법으로 $f(x_0,x_1)=x_0^2+x_1^2$의 최솟값 구하기

def function_2(x):
    return x[0]**2+x[1]**2

init_x=np.array([-3.0,4.0])
gradient_descent(function_2, init_x, lr=0.1, step_num=100)

#학습률이 너무 큰 경우-큰 값으로 발산
init_x=np.array([-3.0,4.0])
gradient_descent(function_2, init_x, lr=10.0, step_num=100)


#학습률이 너무 작은 경우 - 거의 갱신되지 않음 
init_x=np.array([-3.0,4.0])
gradient_descent(function_2, init_x, lr=1e-10, step_num=100)
#학습률 같이 직접 정해야하는 변수: 하이퍼파라미터 

#실제 기울기를 구하는 코드 코드 구현 
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)#형상이 2*3인 가중치 매개변수 하나를 인스턴스 변수로 가짐-정규분포로 초기화 
    
    def predict(self, x): #예측을 수행
        return np.dot(x, self.W)
    
    def loss(self,x,t): #손실함수 
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss

net = simpleNet()
print(net.W) # 가중치 매게변수 

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p)# 최댓값 인덱스 
t=np.array([0,0,1]) #정답 레이블 
net.loss(x,t)

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)

#lambda를 이용해 간단히 구현 
f = lambda w: net.loss(x, t)
dw = numerical_gradient(f, net.W)
