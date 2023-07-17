#손실함수 - 신경망 성능의 '나쁨'을 나타내는 지표 , 훈련 데이터를 얼마나 잘 처리하지 못하느냐를 나타냄 
import numpy as np

#오차제곱합(SSE)
y=np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]) #신경망이 추정한 값 
t=np.array([0,0,1,0,0,0,0,0,0,0]) #정답 레이블/ 원-핫 인코딩 

#오차제곱합 SSE
def SSE(y,t):
    return 0.5*np.sum((y-t)**2)

SSE(y,t)#예상 답: 2, 답:2
#0.0975
y=np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]) #예상 답: 7
SSE(y,t)
#0.5975
#첫번째가 손실함수 출력이 더 작음- 오차가 적음  - 정답에 더 가까움 

#교차엔트로피 CEE
def CEE(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) #np.log()함수에 0을 대입하면 -inf가 되어 계산불가 그래서 아주 작은 값 delta를 더함  

y=np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0])
t=np.array([0,0,1,0,0,0,0,0,0,0])
CEE(y,t)
# 0.51082545709933802
y=np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])
CEE(y,t)
# 2.3025840929945458


# ### 미니배치
# 기계학습: 훈련데이터에대한 손실함수의 값을 최대한 줄이는 매게변수를 찾아냄 - 모든 데이터의 손실함수 합을 구함 - 평균을 구홤 
#미니배치: 데이터 중 일부만 골라 학습 
#지정한 수의 데이터를 무작위로 추출 

#MNIST데이터 호출 
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

#무작위 10장만 산출
train_size = x_train.shape[0] #60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) #0~60000중 10개 무작위 추출
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

#np.random.choice 사용 
np.random.choice(60000,10)

#배치를 지원하는 교차엔트로피 CEE

def CEE(y,t):
    if y.ndim == 1: 
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7))/batch_size #배치 사이즈로 나눠 정규화 

#원-핫 인코딩이 아닐 때 미니배치 교차엔트로피

def CEE(y,t):
    if y.ndim ==1:
        t = t,reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np,log(y[np.arange(batch_size, t)]+1e-7))/batch_size

