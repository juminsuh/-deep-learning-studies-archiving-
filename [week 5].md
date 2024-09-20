# 5주차

태그: 완료

## 더 많은 층의 심층 신경망

- 신경망의 층을 셀 때는 은닉층과 출력층의 개수만 고려한다.
- 얼마나 깊은 신경망을 사용해야 하는지 미리 예측하기 어렵다.

![Untitled](https://github.com/user-attachments/assets/b77bb31f-c713-416c-a9ae-877725461b03)

- $L = 4$ (층의 개수)
- $n^{[l]}$ = 각 층의 노드 개수
    - $n^{[1]} = 5$, $n^{[2]} = 5$, $n^{[3]} = 3$, $n^{[4]} = 1$, $n^{[0]} = n_x = 3$
- $a^{[l]}$ = 층 $l$에서의 활성화 값
- $a^{[0]} = X$, $a^{[L]} = \hat{y}$

## 심층 신경망에서의 정방향 전파

- 일반적인 정방향 전파 수식 - $l$층에 대하여
    - $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$
    - $a^{[l]} = g^{[l]}(z^{[l]})$
- 벡터화된 정방향 전파 - $l$층의 $m$개의 데이터셋에 대하여
    - $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$
    - $A^{[l]} = g^{[l]}(Z^{[l]})$
    - $l = 1, 2, \ldots, L$ (L은 층의 개수) 만큼 for-loop를 사용해 반복한다.

## 행렬의 차원을 알맞게 만들기

![Untitled 1](https://github.com/user-attachments/assets/1664ca64-cef6-4ca2-b2b0-fc51d7d45559)

- $L = 5$인 심층 신경망이다.
- $z^{[l]} = (n^{[l]}, 1)$
- $Z^{[l]} = (n^{[l]}, m)$: $z^{[l]}$ 열 벡터로 이루어짐
- $W^{[l]} = (n^{[l]}, n^{[l-1]}) = dW^{[l]}$
- $a^{[l-1]} = (n^{[l-1]}, 1)$
- $A^{[l-1]} = (n^{[l-1]}, m)$: $a^{[l-1]}$ 열 벡터로 이루어짐
- $b^{[l]} = (n^{[l]}, 1) = db^{[l]}$: $m$개의 데이터셋이 있으면 차원은 $(n^{[l]}, m)$이 됨

## 왜 심층 신경망이 더 많은 특징을 잡아낼 수 있을까요?

![Untitled 2](https://github.com/user-attachments/assets/a8b5f368-3e7b-4fdc-9975-f93d9e735d3c)

- 첫 번째 은닉층에서는 수직 방향의 모서리와 수평 방향의 모서리를 찾는다.
- 두 번째 은닉층에서는 첫 번째 은닉층의 결과를 이용해 눈, 코, 입과 같은 얼굴의 특징을 찾는다.
- 세 번째 은닉층에서는 두 번째 은닉층의 결과를 이용해 얼굴을 인식한다.

**1️⃣ 심층 신경망이 깊어질수록** 낮은 레이어에서는 간단한 특징을 , 높은 레이어에서는 복잡하고 다양한 특징을 많이 추출할 수 있다. 

**2️⃣ 순환 이론에 따르면,** 은닉층이 얼마 없다면 은닉층의 노드 개수가 기하급수적으로 많아져야 한다. → $2^{(n-1)}$개의 노드가 필요하다. 여러 개의 은닉층이 있다면 필요한 노드의 수는 $\log n$이지만, 은닉층이 얼마 없다면 하나의 은닉층에 $2^{n}$개의 노드가 필요하다.

## 심층 신경망 네트워크 구성하기

![Untitled 3](https://github.com/user-attachments/assets/391dba88-1036-4909-bb70-af250fbd6066)

- 정방향 전파의 레이어 $l$에서 $a^{[l-1]}$을 입력으로 받아 $a^{[l]}$을 출력으로 내놓는다. 이 과정에서 $z^{[l]}, W^{[l]}, b^{[l]}$값을 캐시값(cache)으로 저장한다.
- 파이토치와 같은 딥러닝 프레임워크에는 cache를 저장하는 기능이 있지만 일반적으로 python에서는 리스트에 따로 저장해야 한다.
- 역방향 전파는 $da^{[l]}$와 $z^{[l]}$을 입력으로 받아 $da^{[l-1]}$을 출력으로 내놓는다. 이때 $dz^{[l]}, dW^{[l]}, db^{[l]}$도 계산한다.
- 전체적 흐름은 아래 그림과 같다.
- 이후 $dW^{[l]}$와 $db^{[l]}$으로 w와 b를 업데이트한다.

![Untitled 4](https://github.com/user-attachments/assets/5a57c9dd-897a-4538-a599-38f3535ff157)

## 정방향전파와 역방향전파

- 정방향전파 for layer $l$
    - input $a^{[l-1]}$
    - output $a^{[l]}$
    - 한 번에 한 개의 데이터셋: $z^{[l]}=W^{[l]}*a^{[l-1]}+b^{[l]}$
    - 한 번에 m개의 데이터셋(vectorized): $Z^{[l]}=W^{[l]}*A^{[l-1]}+b^{[l]}$
    - $z^{[l]}, W^{[l]}, b^{[l]}$의 값을 **캐시**로 저장해둔다.
    - 가중치 $W^{[l]}$은 행 벡터로 이루어져 있으며, 역방향 전파에 의해 값이 업데이트된다.
    
    ![IMG_5109](https://github.com/user-attachments/assets/6e8f7c34-11a1-408d-a33c-2bf0b1ce66fa)
    
- 역방향 전파 for layer $l$
    - input $da^{[l]}, z^{[l]}$
    - 역방향 전파의 계산에서 전방향 전파 때 저장해두었던 캐시$(z^{[l]}, W^{[l]}, b^{[l]})$를 사용한다.
    
    ![Untitled 5](https://github.com/user-attachments/assets/023db69d-11cf-46b9-8973-900b4ff84e71)
    
    ![Untitled 6](https://github.com/user-attachments/assets/ac2f0259-c7b4-4f37-98c0-a192ec667706)
    
    - 정방향 전파는 입력값 $X$로 값을 초기화한다.
    - 역방향 전파는 $da^{[l]}$로 값을 초기화한다. $dA^{[l]}$은 벡터화한 것으로, $da^{[1]} \ldots da^{[m]}$을 모두 더한 값으로 구할 수 있다.

## 변수 vs 하이퍼파라미터

- parameters: w,b
- hyperparameter: learning rate $\alpha$, #iterations, #hidden layers, #hidden units(node), choice of activation function, momentum term(concept of velocity to the parameter update), mini-batch size
- hyperparameter로 최종 모델의 변수 parameter를 통제할 수 있다.
- applied deep learning is a very **empirical** process→time consuming, hard, variable trials are required.
- 하이퍼 파라미터는 결정된 값이 없으며 여러 번의 시도를 통해 적절한 하이퍼파라미터를 찾아야 한다.

## 인간의 뇌와 어떤 연관이 있을까요?

- 뇌와 딥러닝을 연관지어 설명하는 것은 적절하지 않을 수 있다. 그렇지만 직관적으로 이해하거나 상상력을 자극하기에는 좋은 비유다.
- 하나의 뉴런과 단일 로지스틱 신경망이 대응된다. 입력을 받은 후 활성화 함수를 거친 값이 특정 값 이상이면 신호가 활성화된다는 점이 유사하다.

## QUIZ

![Untitled 7](https://github.com/user-attachments/assets/2fbeead8-a10a-40f2-9eac-bb34b63decd4)

- $a^{[l]}$는 활성화 함수가 아니라 활성화 함수가 적용된 값이기 때문에 하이퍼 파라미터가 아니다.
