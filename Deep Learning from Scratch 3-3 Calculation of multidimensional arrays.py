#다차원 배열
import numpy as np
A = np.array([1,2,3,4])
print(A)
#[1 2 3 4]
#Dimensions of the array
np.ndim(A) 
# 1 
A.shape #Return a tuple/ 배열의 형상 확인 
# (4, )
A.shape[0]
# 4

# 2차원 배열 
B = np.array([[1,2],[3,4],[5,6]])
print(B)
np.ndim(B)
2
B.shape
(3,2)

# 행렬의 곱 
A=np.array([[1,2],[3,4]])
A.shape
# (2,2)
B=np.array([[5,6],[7,8]])
B.shape
# (2,2)
np.dot(A,B) #입력이 1차원 배열이면 벡터를, 2차원 배열이면 행렬 곱을 계산 
np.dot(B,A) #주의// np.dot(A,B) != np.dot(B,A) 

A=np.array([[1,2,3],[4,5,6]])
A.shape
# (2,3)
B=np.array([[1,2],[3,4],[5,6]])
B.shape
# (3,2)
np.dot(A,B)

C=np.array([[1,2],[3,4]])
C.shape
# (2,2)
np.dot(A,C)
# 오류 출력 
np.dot(C,A)

A=np.array([[1,2],[3,4],[5,6]])
A.shape
# (3,2)
B=np.array([7,8])
B.shape
# (2, )
np.dot(A,B)

#scala product in neural network
X=np.array([1,2])
X.shape
W=np.array([[1,3,5],[2,4,6]])
print(W)
W.shape
Y=np.dot(X,W)
print(Y)
