# 넘파이 가져오기 
import numpy as np

#넘파이 배열 생성 
x = np.array([1.0,2.0,3.0])
print(x)
#[1, 2, 3]
type(x)
#<class 'numpy.ndarray>
#np.arry: 리스트 인수, 넘파이 배열(numpy.ndarray) 반환 

#넘파이 산술 연산 
x=np.array([1.0,2.0,3.0])
y=np.array([2.0,4.0,6.0])
#x,y원소 수 같음 >> 각 원소에대해 산술 행해짐 

x+y #원소별 덧셈 
x-y #원소별 뺄셈
x*y #원소별 곱 (element-wise product)
x/y
x/2.0 #배열과 스칼라 계산(브로드케스트)

#넘파이의 N차원 배열 

#2차원 배열 
A = np.array([[1,2],[3,4]])
print(A)
A.shape #행렬의 형상: 각 차원의 크기(원소 수) 
#(2,2) 2x2 크기의 배열 
A.dtype #행렬에 담긴 원소의 자료형 
#dtype('int64')

#N차원 배열 사칙연산
B=np.array([[3,0],[0,6]]) #2차원 배열
A+B
A*B #원소 수 같을 떄 대응하는 원소별로 계산 
print(A)
A*10 #스칼라 산술도 가능 

#브로드케스트: 형상이 다른 배열끼리의 계산 
A=np.array([[1,2],[3,4]])
B=np.array([10,20])
A*B #B가 2차원 배열로 확장되어 연산됨 

#원소 접근
X=np.array([[51,55],[14,19],[0,4]])
print(X) 
X[0] # 0행
#array([51, 55])
X[0][1] #(0, 1)위치의 원소 
#55

#for 문으로 각원소에 접근 
for row in X:
    print(row)

#인덱스를 배열로 지정: 한번에 여러 원소에 접근     
X=X.flatten() #1차원 배열로 평탄화
print(X)
# [51 55 14 9 0 4]
X[np.array([0,2,4])] #인덱스가 0,2,4인 원소 얻기
#array([51, 14, 0])

#특정 조건을 만족하는 원소 얻기 
X>15 #결과 bool 배열 
X[X>15]




