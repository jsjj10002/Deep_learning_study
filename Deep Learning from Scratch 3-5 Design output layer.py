# 출력층 설계 / 회귀: 항등 함수, 분류: 소프트 맥스 함수 사용

#소프트 맥스 함수 구현 
import numpy as np
a = np.array([0.3,2.9,4.0])

exp_a = np.exp(a) #지수함수 
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a/sum_exp_a
print(y)

#implement softMax function 

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    
    return y
# 소프트 맥스 함수 구현 시 오버플로 문제를 주의 해야함
# 오버플로우 : 지수 함수를 포함해서 값이 매우 커지고 이런 값으로 나눗셈을 하면 수치가 불안정해짐

#소프트 맥스 함수 개선 
a = np.array([1010, 1000, 900])
np.exp(a)/np.sum(np.exp(a)) # 소프트맥스 계산 
# array(nan, nan, nan]) #제대로 계산되지 않음 
c = np.max(a)
a-c
np.exp(a-c)/np.sum(np.exp(a-c)) #입력 신호를 최댓값으로 빼줌 

#Softmax function improves overflow problem
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #오버플로우 대책 
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    
    return y

# 소프트맥스 함수의 특징 
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
#[0.01821127 0.24519181 0.73659691] - 0에서 1.0사이의 실수 
np.sum(y)
# 출력의 총합이 1 : 확률적 결론 낼 수 있음 
# 각 원소의 대소 관계의 변화가 없음 - 분류 문제에서 소프트맥스 함수를 생략 일반적 

