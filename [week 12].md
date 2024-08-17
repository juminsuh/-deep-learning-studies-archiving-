# 12주차

태그: 완료

## Softmax Regression

### Softmax Regression

- softmax는 여러 개의 클래스를 분류할 시 사용된다.
- 예를 들어 cats, dogs, baby chicks의 3개의 클래스가 있고, 각각을 1, 2, 3 클래스라고 하자. (셋 중 아무것도 해당되지 않은 사진 other은 0 클래스로 분류한다)

![1db3cff1-cdff-436d-9e1c-f896eb042225](https://github.com/user-attachments/assets/c4a4b41d-3ea6-49d3-a642-84bc3a3193c5)

- 아래와 같은 NN이 있다면, 마지막 층의 각 노드는 input X가 i번째 클래스에 속할 확률 (probability) 이다.

![IMG_5317](https://github.com/user-attachments/assets/fd2d33fd-2e97-443d-9606-38dc35d8b97a)

- NN의 마지막 층은 소프트맥스 층이다.

### Softmax Activation Function

- 소프트맥스의 층의 활성화 함수는 다음과 같이 정의된다.

![IMG_5318](https://github.com/user-attachments/assets/a7acde5c-cd48-4d3f-9515-e039f125120a)

![IMG_5319](https://github.com/user-attachments/assets/a4a76e78-abdd-4815-a3b7-6e5f0bb348d1)

![IMG_5320](https://github.com/user-attachments/assets/9bce4310-845d-4eec-9b02-ef2cb49cc8e6)

- 특징은 실수를 입력으로 받아 실수를 출력하는 다른 활성화 함수(sigmoid, relu)와는 다르게 벡터 (이 문제에서는 (4, 1) 벡터)를 입력 받아 벡터를 출력한다는 점이다.  이는 정규화를 하기 위함이다.
    - binary classification은 L층에서 실수를 받고, 그 실수가 threshold보다 작다면 0, 크다면 1로 분류되는 메커니즘이다.
    - softmax regression은 L층에서 실수가 아니라 벡터를 받는데, 이는 클래스가 3개 이상이기 때문이다. 하나의 실수로 0 / 1을 결정하는 것이 아니라 각 클래스별 확률을 계산하고, 그 중에서 가장 큰 확률을 가진 클래스를 예측 클래스로 사용한다. 이때 정규화를 하는 것은 모든 확률의 합을 1로 만들기 위함이다. 즉, ‘정규화하기 위해 벡터를 입출력으로 받는다’는 표현은 각각의 클래스의 확률을 나타내기 위함이라는 말과 같다.
    
    <img width="731" alt="Untitled" src="https://github.com/user-attachments/assets/aa1b03e1-f20d-4826-9b1a-5ae0121401a4">
    

### Softmax Examples

- 은닉층이 없고, C = 3일 때 입력 특성 x1, x2에 따라 data의 클래스를 softmax regression으로 분류한 예시
- 2번째 그래프가 제일 잘 분류했고 1, 3번째는 약간의 오차가 보인다.

![Untitled 1](https://github.com/user-attachments/assets/945248bc-968e-41b7-a90c-29f329d2058d)


- 얻을 수 있는 직관: 두 클래스 사이의 경계가 선형이다.
- 만약 여러 개의 은닉층을 추가한다면 더 복잡한 비선형의 경계도 볼 수 있다.

## Softmax 분류기 훈련시키기

### Softmax Regression의 특징

- hardmax: softmax의 반대 개념
    - softmax: z 벡터 → 활성화 함수 g → a 벡터 → z 벡터의 값을 부드러운 느낌으로 각각의 클래스일 확률로 대응
    - hardmax: z 벡터 중 가장 큰 값이 있는 곳에 1, 나머지는 0을 할당 → 아주 단호한 느낌
- softmax regression generalizes logistic regression to C classes
    - C = 2이면 logistic regression과 같다. C = 2인 softmax regression은 두 개의 확률을 내놓는데, 어차피 확률의 합이 1이므로 하나만 계산해도 된다. 이는 결국 y = 1일 확률을 계산하는 logistic regression과 똑같은 흐름이.
    
    ![Untitled 2](https://github.com/user-attachments/assets/9d08e8a4-0d03-4917-9031-dddb03634c09)
    

### Loss function

<img width="748" alt="Untitled 3" src="https://github.com/user-attachments/assets/be587f8b-cb6f-40cf-b941-516cb1367c92">

- 일반적으로 loss function은 예측 확률에 따른 예측 클래스가 뭐든 간에 정답 클래스에 대응하는 확률을 가능한 한 크게 만드는 것이 목표다.

### Cost function

- cost function J는 전체 데이터의 loss를 구한 뒤 전체 데이터의 개수 m으로 나눠줘서 구할 수 있다.

<img width="790" alt="Untitled 4" src="https://github.com/user-attachments/assets/118fd438-25dc-4b6a-82e5-a6a2b7a497aa">

### Gradient Descent with softmax

![Untitled 5](https://github.com/user-attachments/assets/9c0c85a0-282a-4a78-b9ad-b0a8736c7727)


<img width="731" alt="Untitled 6" src="https://github.com/user-attachments/assets/c3fd22e3-925b-46d3-9f23-dc3c22a7b892">

![Untitled 7](https://github.com/user-attachments/assets/d95f85cf-abf8-4fc3-8650-af4a17be1990)

- 딥러닝 프레임워크: forward propagation하는 법을 정하면 프레임워크가 스스로 미분 계산을 통해 backpropagation을 해준다.

## Deep Learning Frameworks

![Untitled 8](https://github.com/user-attachments/assets/9ce574ca-77eb-40f2-97ae-579eac2df80d)


### choosing deep learning frameworks

1. Ease of Programming: development and deployment(전개, 상용화)
2. running speed
3. truly open: open source with good governance → 오픈 소스이며 잘 관리되고 있는지, 혹은 오픈 소스였던 것을 폐쇄할 가능성이 있는지. 즉, 신뢰도의 문제이다. 

## Tensorflow

** tensorflow 2.x를 기준으로 코드를 다시 작성함**

![Untitled 9](https://github.com/user-attachments/assets/708bb097-2ae0-42fc-8e18-8ee441b4a0c9)

- optimizer.apply_gradients(zip(gradients, [w])): gradients와 [w]를 반복 가능한 객체(iterator)로 만들어 gradients와 [w]의 요소를 각각 하나씩 가져와 튜플 형태로 반환하는 것을 반복한다.

![Untitled 10](https://github.com/user-attachments/assets/36544f14-ad05-401e-bd29-0f073f8c7021)


- 강의에서 사용하신 tf.placeholder는 버전이 2.x로 바뀌면서 사라졌다. x에 어떤 coefficient가 주어지냐에 따라 최소화해야 하는 cost function이 달라지기 때문에 이에 따른 최적의 w가 달라질 수 있다. 이 예시에서는 위와 같이 1, 10, 25를 계수로 썼기 때문에 같은 결과가 나온다.

![Untitled 11](https://github.com/user-attachments/assets/27d947fe-8f01-4068-9f77-a85b4550b688)

- tensorflow에서 비용 함수를 명시해주면, 자동으로 미분을 계산하고 비용 함수를 최소화해준다. 즉, 정방향 전파를 정의하는 것은 역방향 전파 함수도 구현한 것과 같다.
