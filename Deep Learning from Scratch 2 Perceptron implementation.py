#Initialize 'AND logic' function

def AND(x1,x2): # x1, x2를 인수로 받는 AND함수
    w1, w2, theta=0.5,0.5,0.7 #매게변수 초기화 
    tmp = x1*w1+x2*w2 #가중치를 곱한 입력의 총합이 임계값을 넘으면 1 반환, 나머지 0 반환 
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

# AND 게이트 
print(AND(0,0)) # 0출력
print(AND(1,0)) # 0출력
print(AND(0,1)) # 0출력
print(AND(1,1)) # 1출력

# 가중치와 편향(bias) 도입 

import numpy as np
x = np.array([0,1])#입력
w = np.array([0.5, 0.5])#가중치
b = -0.7
w*x
np.sum(w*x)
np.sum(w*x)+b

#implement AND gate with bias
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5, 0.5]) # 가중치: 신호가 결과에 주는 중요도를 조절하는 매게변수 
    b = -0.7 #theta를 -b로 치환 / 편향: 뉴런이 얼마나 쉽게 활성화(1 출력)하는지 조절하는 매게변수 
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#impelement NAND gate
def NAND(x1, x2):
    x=np.array([x1,x2])
    w=np.array([-0.5, -0.5])
    b=0.7
    tmp = np.sum(w*x)+b
    if tmp <=0:
        return 0
    else:
        return 1
    
#impelement OR gate
def OR(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b= -0.2
    tmp= np.sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1
    
#impelement XOR gate (배타적 논리합: 한쪽이 1 일때만 1 출력)
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

print(XOR(0,0)) # 0을 출력
print(XOR(1,0)) # 1을 출력
print(XOR(0,1)) # 1을 출력
print(XOR(1,1)) # 0을 출력 

#XOR: 2층 퍼셉트론 
