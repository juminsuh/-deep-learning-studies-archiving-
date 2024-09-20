# 2주차

태그: 완료
텍스트: Kaggle 필사

## 01. 계산 그래프

- $J(a,b,c)=3(a+bc)$의 계산 그래프 만드는 과정-왼쪽에서 오른쪽으로 전방향 전파(forward pass)를 거쳐  $J(w,b)$  비용 함수를 계산


1. $u=bc$
2. $v=a+u$
3. $J=3v$

![Untitled](https://github.com/user-attachments/assets/ad2cb633-bd4a-4c07-ad32-b98792bd23c3)

## 02. 계산 그래프로 미분하기

- 미분의 연쇄법칙(chain rule)이란 합성함수의 도함수에 대한 공식
- 합성함수의 도함수(derivative)는 합성함수를 구성하는 함수의 미분을 곱함으로써 구할 수 있음(겉미분*속미분)
- 기본적인 아이디어: 오른쪽에서 왼쪽으로 역전파(backpropagation) 진행 → $x$가 바뀜에 따라 $J$는 어떻게 바뀌는가? → $\frac{dJ}{dx}$

    - $v=a+u→J=3v$
    - $\frac{dJ}{da}=\frac{dJ}{du} \frac{du}{da}$
- 코드 작성 시 편의를 위해 아래와 같이 도함수를 정의함
    - 최종변수를 Final output var, 미분하려는 변수를 var이라고 정의
    
    $$
    \frac{d \text{Final output var}}{d \text{var}} = d \text{var}
    $$
    
- 위의 예시로 계산한 결과

$$
dv=\frac{dJ}{dv}=3
$$

$$
du=\frac{dJ}{dv}\frac{dv}{du}=3*1=3
$$

$$
da=\frac{dJ}{dv}\frac{dv}{da}=3*1=3
$$

$$
db=\frac{dJ}{du}\frac{du}{db}=3*2=6
$$

$$
dc=\frac{dJ}{du}\frac{du}{dc}=3*3=9
$$

## 03. 로지스틱 회귀의 경사하강법

- 단일 샘플에 대한 경사하강법

---

![Untitled 1](https://github.com/user-attachments/assets/f61515fe-7378-4753-a145-cd0eb9c7d15a)

$$
da=\frac{dL(a,y)}{da}=-\frac{y}{a}+\frac{1-y}{1-a}
$$

$$
dz=a-y
$$

$$
dw_{1}=\frac{dL}{dw_{1}}=x_{1}dz, dw_{2}=\frac{dL}{dw_{1}}=x_{2}dz
$$

$$
db=\frac{dL}{db}=dz
$$

➡️ **결과**

$$
w_{1}=w_{1}-\alpha dw_{1}, w_{2}=w_{2}-\alpha dw_{2}, b: b-\alpha db
$$

## 04. m개 샘플의 경사하강법

- 단일 샘플이 아닌 **m개의 샘플**에 대한 경사하강법
- 로지스틱 회귀에서 비용 함수는 다음과 같이 표현됨

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{i=m}(L(a^{(i)},y^{(i)})) when (x^{(i)},y^{(i)})
$$

- **코드**

![IMG_5078](https://github.com/user-attachments/assets/a56872ac-1879-47ca-911d-f53a03be4e24)

Details

$dw_{1}, dw_{2}, db$는 값을 저장하는 데 사용되고 있음. 이 식들는 첨자 $(i)$가 사용되지 않는데, 이는 식들이 훈련 세트 전체를 합한 값을 저장하고 있기 때문임. 반면 $dz^{(i)}$는 훈련 샘플 하나 당$(x^{(i)},y^{(i)})$의 $dz$이기 때문에 첨자 $(i)$를 사용함.

➡️ 위 코드를 한 번 실행하면 한 단계의 경사 하강법을 실행하는 것. 따라서 훈련을 위해선 경사 하강법을 여러 번 실행해야 함.

**그러나, 이 방법엔 문제가 있음**

for 문을 두 개 써야 한다는 점. 첫 번째 for문은 m개의 샘플 데이터를 도는 데, 두 번째 for 문은 n개의 특성을 도는 데 쓰임→이런 명시적인 for문은 알고리즘을 비효율적으로 만듦(계산 속도가 느려짐)→**명시적인 for문 없이 코드를 구현해야 큰 데이터 집합도 처리 가능!**

✅ **vectorization(벡터화): 파이썬의 내장 함수를 이용하여 명시적인 for문을 제거, 큰 데이터 집합을 용이하게 처리하는 방법**

## 05. 벡터화(vectorization)

- 벡터화의 예시

![IMG_5085](https://github.com/user-attachments/assets/61507690-5e8f-4d68-8ba3-b36860f0c9a4)

- SIMD(Single Instruction Multiple Data): 병렬 프로세서의 한 종류로, 하나의 명령어로 여러 개의 값을 동시에 계산하는 방식. 이는 벡터화 연산을 가능하게 함. CPU와 GPU를 이용한 계산에 모두 적용할 수 있음.

## 06. 더 많은 벡터화 예제

- 컴퓨터의 계산 효율성을 위해서 가능하면 for문을 피하는 것이 좋음
- 벡터화를 위해 자주 쓰는 numpy 함수
    - np.dot(a,b) # inner product
    - np.exp(v) # exponential
    - np.log(v) #log v
    - np.abs(v) #absolute value
    - np.maximum(v,0) # v와 0중 큰 값을 반환
    - ** #squared value
    - 1/v #inverse of v
    - np.zeros(m,n) # (m,n) 짜리 0 행
- 로지스틱 회귀에는 두 개의 for문이 존재(m개의 훈련 데이터셋 학습/n개의 특징 업데이트)→아래 코드는 n개의 특징을 업데이트하는 for문을 벡터화로 대체한 예

![IMG_5080](https://github.com/user-attachments/assets/6ed459a3-f95b-4201-a263-2b43b633c908)

## 07. 로지스틱 회귀의 벡터

- 벡터화를 통해 m개의 샘플의 forward pass를 동시에 계산하는 방법-for문을 아예 쓰지 않음
    - 원래대로라면 아래의 식은 for문을 통해 i를 변화시켜 가며 계산해야 했음
        - $z^{(i)}=w^{T}x^{(i)}+b$
        - $a^{(i)}=\sigma(z^{(i)})$
    - 하지만 벡터화를 사용하면 다음과 같이 간결하게 계산할 수 있음
        $$
        Z = [z^{(1)}, z^{(2)}, \ldots, z^{(m)}] = \text{np.dot(np.transpose(W), X)} + b
        $$        
        
        ![IMG_5077](https://github.com/user-attachments/assets/eccb91f2-aab8-44e9-9d26-1d1003181251)
        
        ✅ $np.dot(np.transpose(w), x)$는 (1,m)크기의 행렬과 상수 b를 더해 오류가 날 것 같지만, 파이썬이 자동적으로 상수 b를 (1,m) 크기의 행렬로 **브로드캐스팅** 해주기 때문에 오류가 발생하지 않음
        
        - $A=[a^{(1)}, a^{(2)}…a^{(m)}]=\sigma(Z)$

## 08. 로지스틱 회귀의 경사 계산을 벡터화 하기

- 벡터화를 통해 m개의 샘플에 대한 경사 계산을 동시에 하는 방법

![IMG_5083](https://github.com/user-attachments/assets/a38e3820-ce61-4d9a-bdb6-b854c01d995d)

![IMG_5084](https://github.com/user-attachments/assets/d68ea90c-918c-48c2-8f0b-4c50bee54c8d)

그러나 경사 하강을 여러 번 한다면, 이때는 어쩔 수 없이 for문을 써야 함



## **Quiz**

1. **로지스틱 회귀에서 벡터화가 중요한 이유**
    
    -벡터화는 수학적 연산의 효율을 높여주는 것이다.
    
    -Numpy와 같은 고수준 라이브러리를 이용하면 하드웨어의 가속을 통해 큰 효율을 볼 수 있다. 
    
    -벡터화는 벡터화 되지 않은 것보다 빠른 계산이 가능하다. 
    



## 캐글 필사

## 00. file ready

```python
#kaggle/working/input 디렉토리 미리 만들기
#kaggle에서 X.npy, Y.npy 다운로드
#코랩에서 하려면 그냥 /content/drive/MyDrive/X.npy 이런 식으로 하기

from google.colab import drive
drive.mount('/content/drive')

# 작업 디렉토리 설정-원래 /kaggle/input은 읽기 전용이기 때문에 /kaggle/working/input/과 같이 쓰기 파일을 만들어줘야 함

import os

os.chdir('/kaggle/working/')

#/content/drive/MyDrive/에 있던 X.npy와 Y.npy를 /kaggle/working/input/으로 옮김

import shutil

x_source_file_path = '/content/drive/MyDrive/X.npy'
x_destination_folder_path = '/kaggle/working/input/'
shutil.move(x_source_file_path, x_destination_folder_path)

y_source_file_path = '/content/drive/MyDrive/Y.npy'
y_destination_folder_path = '/kaggle/working/input/'
shutil.move(y_source_file_path, y_destination_folder_path)
```

## 01. Introduction

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # 경고 메시지 무시
from subprocess import check_output
print(check_output(["ls", "/kaggle/working/input"]).decode("utf8")) #'ls' 명령어를 실행해 디렉토리의 파일 목록을 프린트. 이때 decode("utf8")은 바이트 문자열을 일반 문자열로 디코딩
```

## 02. Overview the Dataset

*   간편성을 위해 이 튜토리얼에서는 오직 0과 1만을 사용
*   데이터에서, 0의 index는 204부터 408으로 총 205개가 있고 1의 index는 822부터 1027으로 총 206개가 있기 때문에 총 205개의 샘플을 사용할 것
*   X array: image array(0 or 1)
*   Y array: label array(0 or 1)

```python
#load dataset

x_loader=np.load('/kaggle/working/input/X.npy')
Y_loader=np.load('/kaggle/working/input/Y.npy')

img_size=64

plt.subplot(1,2,1) # 1행 2열의 1번째 자리
plt.imshow(x_loader[260].reshape(img_size, img_size)) # x_loader의 260번째 자리에 저장된 이미지(0)를 표시. 이미지의 크기를 재구성
plt.axis('off') # 서브플롯의 축을 없앰

plt.subplot(1,2,2) #1행 2열의 2번째 자리
plt.imshow(x_loader[900].reshape(img_size, img_size), cmap='gray') # x_loader의 900번째 자리에 저장된 이미지(1) 표시. 이미지의 크기를 재구성
plt.axis('off') # 서브플롯의 축을 없앰

#밑의 결과는 이미지 플롯의 축 범위를 나타냄-(xmin, xmax, ymax, ymin)

#sign 0이 205개이므로 sign 1도 205개로 맞춰주느라 1028이 아니라 1027임 
X=np.concatenate((x_loader[204:409], x_loader[822:1027]), axis=0) # concatenate zero sign and one sign in order to create image array in this tutorial

z=np.zeros(205) # create 0 sign label array
o=np.ones(205) # create 1 sign label array
Y=np.concatenate((z,o),axis=0).reshape(X.shape[0],1) #열벡터로 만듦 

print("X shape:", X.shape) #(410,64,64)
print("Y shape:", Y.shape) #(410,1)
```

*   In X.shape=(410,64,64), 410 means the number of images(zero and one signs). 64X64 means the size of a image
*   In Y.shape=(410,1), 410 means the number of labels(0 and 1).
*   Let's split X and Y into trainset and testset
*   test=15% and train=75%
*   test_size=percentage of testset
*   random_state=use **same seed** while randomizing. Setting seed means if we call train_test_split repeatedly, it'll always create the same train and test distribution

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.15, random_state=42)

number_of_train=X_train.shape[0]
number_of_test=X_test.shape[0]
#두 개 더하면 410개
#X_train.shape-(348, 64,64)
#X_test.shape-(62,64,64)
#Y_train.shape-(348, 1)
#Y_train.shape-(62,1)

print(number_of_train) #348
print(number_of_test) #62
```

*   since we have 3D input array(X), so we need to make it flatten to 2D in order to use it as a input for deep learning model
*   label array(Y) is already 2D, so we leave it like that

```python
X_train_flatten=X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[2])
X_test_flatten=X_test.reshape(number_of_test, X_test.shape[1]*X_test.shape[2])

print("X train flatten:", X_train_flatten.shape) #(348,64*64)
print("X test flatten:", X_test_flatten.shape) #(62,64*64)

#transpose-벡터화(vectorization)

x_train=X_train_flatten.T
x_test=X_test_flatten.T
y_train=Y_train.T
y_test=Y_test.T

print("x train:", x_train.shape) #(64*64, 348)-(n_x,m)
print("x test:", x_test.shape) #(64*64, 62)-(n_x, m)
print("y train:", y_train.shape) #(1,348)-(1,m)
print("y test:", y_test.shape) #(1,62)-(1,m)
```

## 03. Logistic Regression

***  Computation Graph**
    *   Computation graphs are a nice way to think about mathematical expressions
    *   parameters: weight(W), bias(b)
    *   why sigmoid?
        *  it gives probabilistic result
        * it is derivative, so we can use it in gradient descent algorithm
    *   if result(y_hat)=0.9, then it means that classification result is 1 with 90% probability

***   Initializing Parameters**
    *   each pixels has its own weight and is multiplied with their weights
    *   initial weight: 0.01 with (4096,1) # W.shape=(4096,1), W.T.shape=(1,4096)
    *   initial bias: 0

```python
#example of definition

def dummy(parameter):
    dummy_parameter=parameter+5
    return dummy_parameter
result=dummy(3) #result=8

#initialize parameters

def initialize_weights_and_bias(dimension):
    w=np.full((dimension,1),0.01) #값이 모두 0.01로 채워진 (dimension,1)짜리 행렬을 생성하
    b=0.0
    return w,b

w,b=initialize_weights_and_bias(4096)
```

***   Forward Propagation-all steps from pixels to cost**
    *   calculate loss function
    *   cost function is summation of loss function
    * each image creates loss functions
    *   write sigmoid definition that takes z as input parameter and returns y_head

```python
# calculation of z
# z=np.dot(w.T,x_train)+b

def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head

y_head=sigmoid(0)
y_head #0.5

# forward propagation steps
# find z=w.T*x+b
# y_head=sigmoid(z)
# loss=loss(y,y_head)
# cost=sum(loss)

def forward_propagation(w,b,x_train,y_train):
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1] # x_train.shape[1] is for scaling, 아까 x_train을 transpose 해줬기 때문에 x_train.shape[1]은 348, 즉 이미지의 갯수

    return cost
```

***   Optimization Algorithm with Gradient Descent**
    *   in order to decrease cost, we need to update weights and bias that minimize cost->gradient descent
    *   learning rate(alpha): kind of hyperparameters that must be tunned. speed at updating parameters(if learning rate is too small, then the update might take long time. if learning rate is too big, then the update will be fast and it can be crashed)

```python
#combination of forward propagation and backpropagation

def forward_backward_propagation(w,b,x_train, y_train):

    #forward propagatiion
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1]

    #backward propagation
    derivative_weight=(np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1] # dw=X*dZ.T, dZ=y_head-y
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1] #db=np.sum(dZ)
    gradients={"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    return cost, gradients
    
#Updating parameters

def update(w,b,x_train,y_train, learning_rate, number_of_iteration):
    cost_list=[]
    cost_list2=[]
    index=[]

    for i in range(number_of_iteration):
        cost, gradients=forward_backward_propagation(w,b,x_train, y_train)
        cost_list.append(cost)

        #update
        w=w-learning_rate*gradients["derivative_weight"]
        b=b-learning_rate*gradients["derivative_bias"]

        if i%10==0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration {}:{}".format(i,cost))

    # we've finished update parameters(weights and bias)
    parameters={"weight":w, "bias":b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients, cost_list
    
#prediction
#if z is bigger than 0.5->Y_prediction(y_head)=sign 1
#if z is smaller than 0.5->Y_predicti(y_head)=sign 0

def predict(w,b,x_test):

    z=sigmoid(np.dot(w.T,x_test)+b) #(1,348)
    Y_prediction=np.zeros((1,x_test.shape[1])) #(1,348)

    for i in range(z.shape[1]):
        if z[0,i]<=0.5: #z의 [0번째 행, i번째 열]
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1

    return Y_prediction
    

#put all things we've done together

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    #intialize
    dimension=x_train.shape[0] #4096
    w,b=initialize_weights_and_bias(dimension) #weights, bias initialize
    parameters, gradients, cost_list=update(w,b,x_train, y_train, learning_rate, num_iterations)

    y_prediction_test=predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train=predict(parameters["weight"], parameters["bias"], x_train)

    #print train/test loss
    print("train accuracy:{}%".format(100-np.mean(np.abs(y_prediction_train-y_train))*100))
    print("test accuracy:{}%".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))

logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=150)
```

***   Logistic Regression with Sklearn**
    *   in sklearn library, there is a logistic regression method that ease implementing logistic regression

```python
from sklearn import linear_model

logreg=linear_model.LogisticRegression(random_state=42, max_iter=150)
print("test accuracy:{}%".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy:{}%".format(logreg.fit(x_train.T,y_train.T).score(x_train.T, y_train.T)))
```

*   **Summary**
    *   Initialize weights and bias(parameters)
    *   forward propagation
    *   loss function
    *   cost function
    *   backpropagation(gradient descent)
    *   prediction with learnt weights and bias
    *   logistic regression with sklearn
