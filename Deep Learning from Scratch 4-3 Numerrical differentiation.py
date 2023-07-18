#수치미분 구현 
import numpy as np

#미분의 나쁜 구현의 예
def numerical_diff(f,x):
    h = 1e-50 #반올림 오차 (rounding error)문제를 일으킴
    return (f(x+h)-f(x))/h #전방차분
    
np.float32(1e-50)
# 0.0 32비트로 나타내면 오차가 발생 

#두 개선점 적용해 미분 구현 - 1e-4 사용 , 중앙차분: (x+h)와 (x-h)차이의 차분을 계산 

def numerical_diff(f,x):
    h = 1e-4 #0.0001
    return (f(x+h)-f(x-h))/(2*h) #중앙 차분

#수치미분의 예 
# $y= 0.01x^2+0.1x$ 함수 정의 

def function_1(x):
    return 0.01*x**2 +0.1*x
#함수 그리기 
import matplotlib.pylab as plt 

x = np.arange(0.0, 20.0, 0.1) #0.0에서 20.0까지 0.1간격으로 배열 x를 만듦(20 미포함)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

numerical_diff(function_1, 5) # x가 5일때 미분 계산 
# 0.1999999999990898
numerical_diff(function_1, 10) # x가 10일때 미분 계산
# 0.2999999999986347
#오차가 매우 작음 

#구한 수치미분 값을 기울기로 하는 직선 그리기 
def tangent_line(f,x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

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
# $ f(x_0, x_1) = x_0^2 + x_1^2 $ 변수가 2개인 함수 

#함수 구현 
def function_2(x):
    return x[0]**2 + x[1]**2 #인수x : 넘파이 배열
    # 또는 return np.sum(x**2)


#x0=3, x1=4 일때 x0에 대한 편미분 

# 1. x1에 4 대입 
def function_tmp1(x0):
    return x0*x0 + 4.0**2
# 2. x0=3에서 미분 값 구하기 
numerical_diff(function_tmp1, 3.0)

#x0=3, x1=4 일때 x1에 대한 편미분 

def function_tmp2(x1):
    return 3.0**2 + x1*x1

numerical_diff(function_tmp2, 4.0)

