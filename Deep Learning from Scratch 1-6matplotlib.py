#matplotlib: 그래프를 그려주는 라이브러리 

#단순한 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt #matplotlib의 pyplot모듈 이용
#데이터 준비
x=np.arange(0,6,0.1) #0에서 6까지 0.1간격으로 생성 
y=np.sin(x) #sin함수 적용해 y에 할당 

#그래프 그리기 
plt.plot(x,y) #그래프 생성
plt.show() #그래프 화면에 출력

#pyplot 기능

#데이터 준비 
y1=np.sin(x) #sin함수
y2=np.cos(x) #cos함수 

#그래프 그리기
plt.plot(x,y1,label="sin")
plt.plot(x,y2, linestyle="--", label="cos") #cos함수 점선 
plt.xlabel("x") #x축 이름
plt.ylabel("y") #y축 이름 
plt.title("sin & cos") #그래프 제목 
plt.legend()
plt.show()

#이미지 표시 
from matpolotlib.image import imread
img = imread('*') #이미지 읽어오기(결로 설정)
plt.imshow(img)
plt.show()