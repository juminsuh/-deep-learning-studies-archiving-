# 3주차

태그: 완료

## 01. 파이썬에서의 브로드캐스팅


✅ **Broadcasting example**

- for문을 사용하지 않고 각 음식에 포함된 탄수화물/단백질/지방의 비율을 계산하기

![Untitled](https://github.com/user-attachments/assets/05d17bbd-4130-43cf-9356-c88412c77162)

1️⃣ **··· 네 가지 음식에 포함된 칼로리의 합(cal)**

2️⃣ **··· 네 열을 각 열의 합으로 나누기(percentage)**

```python
import numpy as np

A=np.array([[56.0,0.0,4.4,68.0],[1.2,104.0,52.0,8.0],[1.8,135.0,99.0,0.9]])
cal=A.sum(axis=0) # axis=0-세로로 더하라->axis=0(행) 부분을 건드리겠다는 의미. ex) (3,4)를 axis=0(열 별 합)기준으로 계산하면 (1,4)로 바뀜
percentage=100*A/cal.reshape(1,4) #파이썬의 브로드캐스팅의 예시-(3,4) 행렬을 (1,4) 행렬로 나눔
# 각 음식의 탄수화물/단백질/지방의 구성 비율을 (3,4) 행렬로 출력
print(percentage)
```

✅ **General Principle**

![IMG_5086](https://github.com/user-attachments/assets/9f7a26ad-19a3-4a4e-8ee4-0ec26034aed3)

## 02. 파이썬과 넘파이 벡터


✅ **브로드캐스팅의 장단점**

**장점:** 언어의 넓은 표현성과 유연성을 짧은 코드로 해결해 줌

**단점:** 이 유연성은 브로드캐스팅의 자세한 내용과 작동 방법을 모른다면 이상하고 찾기 어려운 오류가 생기기도 함

```python
import numpy as np

a=np.random.randn(5) #가우시안 분포를 따르는 5개의 숫자가 랜덤으로 생성
print(a.shape) #(5,): rank=1(열 벡터도, 행 벡터도 아님); a==a.T
print(np.dot(a,a.T)) # 실수 하나가 나옴

a=np.random.randn(5,1) #(5,1)인 행렬
print(np.dot(a,a.T)) #(5,5) 행렬
```

<aside>
🚨 **python에서 프로그래밍 예제/신경망에서 로지스틱 회귀 코드를 작성할 때 rank=1(n,)인 이상한 배열을 사용하지 않도록 하자→대신 행 벡터/열 벡터를 사용한다면 벡터의 연산을 훨씬 잘 이해할 수 있다**

</aside>

## 03. 로지스틱 회귀의 비용함수 설명


![IMG_5087](https://github.com/user-attachments/assets/c6b8a5ac-b5f3-4c95-8acf-5eb5567bb4c3)

![IMG_5088](https://github.com/user-attachments/assets/d9c887de-60d4-40eb-a78d-f42beafe30ce)

P(y=1|x)←x가 주어졌을 때 y가 1일 확률

P(y=0|x)←x가 주어졌을 때 y가 0일 확률

이때, 두 식을 하나로 합친다는 것은 y=1일 때는 P(y=1|x)가 늘어나고 P(y=0|x)는 줄어들며, y=0일 때는 P(y=0|x)는 늘어나고 P(y=1|x)는 줄어든다는 것을 의미한다.  

## 04. 신경망 네트워크 개요


![IMG_5089](https://github.com/user-attachments/assets/4cee808d-3f4e-4d35-a0a4-76b2bfccbd1a)

## 05. 신경망 네트워크의 구성 알아보기


![Untitled 1](https://github.com/user-attachments/assets/4dd5aef1-43c9-4a4b-815f-915db920201e)

- input layer($a^{[0]}$=$X$)-hidden layer($a^{[1]}$)-output layer$(a^{[2]}=\hat{y}$)
- 은닉층은 입력층과 출력층 사이의 모든 층을 의미하며, 은닉층에 있는 값은 알 수 없음
- $l$번째 은닉층의 n번째 노드는 $a_{n}^{[l]}$으로 표기함
    - 예를 들어 1번째 은닉층의 첫 번째 노드는 $a_{1}^{[1]}$으로 표기, 1번째 은닉층의 두 번째 노드는 $a_{2}^{[1]}$으로 표기
- $a^{[0]}$에서 $a$는 활성값을 의미
- 입력층에서 X가 은닉층으로 넘어가면 $a^{[1]}=(a_{1}^{[1]}, a_{2}^{[1]}, a_{3}^{[1]}, a_{4}^{[1]})$을 내놓음. 이때 은닉층은 파라미터 $w^{[1]}, b^{[1]}$와 관련되어 있음. $w^{[1]}$은 (4,3) 행렬, $b^{[1]}$은 (4,1) 행렬임
- 출력층은 실수  $\hat{y}=a^{[2]}$를 내놓고, 파라미터 $w^{[2]}, b^{[2]}$와 관련되어 있음. $w^{[2]}$은 (1,4) 행렬, $b^{[2]}$은 (1,1) 행렬임
- n번째 층에서 파라미터의 차원은 $w$는 (n번째 노드의 개수, n-1번째 노드의 개수), $b$는 (n번째 노드의 개수, 1)
- 입력층은 층 수를 세는 데 포함되지 않음. 위의 그림은 2층 신경망임

## 06. 신경망 네트워크 출력의 계산


- 하나의 노드에서 두 개의 연산을 거침. $l$번째 층에서 n번째 노드라고 한다면,
    1. $z_{n}^{[l]}=w_{n}^{[l]T}x+b_{n}^{[l]}$
    2. $a_{n}^{[l]}=\sigma (z_{n}^{[l]})$
    
    ➡️ 만약 for문을 통해 이 연산을 해당 층의 모든 노드에 대해 한다면 상당히 비효율적일 것→**벡터화 필요!** 
    
    ![IMG_5091](https://github.com/user-attachments/assets/7f1d81a7-dfc1-415b-a16b-570137c4788c)
    
    이 네 개의 방정식만으로 연산 가능!
    

## 07. 많은 샘플에 대한 벡터화


- $a^{[j](i)}$

    - $j: j$번째 층
    - $i: i$번째 훈련 데이터
- 만약 m개의 데이터셋에 대해 네 개의 연산을 반복하고 싶다면, 모든 z와 a에 첨자 $[i]$를 붙이고 for i in range(1,m+1)문을 돌려야 함→**벡터화 해야 함!**

![IMG_5094](https://github.com/user-attachments/assets/74a19cba-6707-4bfc-ae1b-7dbec8e49c7e)

*행의 개수는 층의 노드의 개수(z.shape=(#node, 1))이고 열의 개수는 데이터셋의 개수(i=1…m)이다.*

---

## Quiz

![Untitled 2](https://github.com/user-attachments/assets/02983851-1f09-4d41-84a5-48ddaab06ac0)

axis=0은 열(세로) 방향으로 더하라는 의미, axis=1은 행(가로) 방향으로 더하라는 의미다. 따라서 B.shape는 (4,)이고 C.shape는 keepdims=True이므로 2차원 배열을 유지하기 때문에 (1,3)이다. 

(4,)은 1차원 행렬이기 때문에 행 벡터나 열 벡터로 취급되지 않는다. 그러나 (,3)은 일반적으로 (1,3)에서 1이 생략된 2차원 행렬이라고 생각한다.
