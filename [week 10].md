# 10주차

태그: 완료
텍스트: Kaggle 필사

## 미니 배치 경사 하강법

- 배치 경사 하강법: 전체 훈련 샘플에 대해 훈련 후 경사 하강 진행
- 벡터화는 m개의 샘플에 대해 계산을 효율적으로 만들어준다.
    
    → 만약 m = 5,000,000개라면? → 느림 
    
- 미니 배치 경사 하강법: 전체 훈련 샘플을 작은 훈련 샘플로 나눈 후, 미니 배치 훈련 후 경사 하강 진행
    
    ⇒ mini batch 사용: mini batch 사이즈가 1,000개라면 5,000개의 mini batch가  있는 것 
    
    <img width="773" alt="Untitled" src="https://github.com/user-attachments/assets/9e66c359-9084-4735-9d66-80d526af77c7">
    
    ```
    for t = 1, 2, ... 5000 # len(X) / self.batch_size 번 반복
    
    	forward propagation of $X^{t}$ # 벡터화된 X
    		Z^{[1]} = W^{[1]}X^{t} + b^{[1]} # X^{t}.shape = (n_x, 1000)
    		A^{[1]} = g(Z^{[1]})
    		...
    		A^{[L]} = g(Z^{[L]})
    		
    	compute cost J^{{t}} = 1/1000$\sum_{i=1}^{1000}(\hat{y^{(i)}}, y^{(i)})$ + \lambda/2*1000*frobenius norm^2
    	
    	backpropagation to compute gradient of J^{{t}}
    	W^{[L]}:= W^{[L]} - \alpha dW^{[L]}, b^{[L]}:= b^{[L]} - \alpha db^{[L]}
    ```
    

## 미니 배치 경사 하강법 이해하기

- epoch: 한 번의 훈련 과정
- 배치 경사 하강법: 한 번의 경사 하강 훈련 / 미니 배치 경사 하강법: 5000번의 경사 하강 훈련
- 배치 경사 하강법에서는 비용 함수 J가 계속 감소한다.
- 미니 배치 경사 하강법에서는 매 반복마다 다른 훈련 세트에서 훈련되기 때문에 전체적으로 비용 함수 J는 감소하나 약간의 노이즈가 발생한다. 예를 들어 X^{1}, Y^{1}은 훈련하기 쉬운 미니배치여서 비용이 약간 낮은데, X^{2}, Y^{2}이 훈련하기 어려운 미니배치라면 비용이 높아질 수 있다. 그러나 전체적으로 비용 함수 J는 감소한다.

![Untitled 1](https://github.com/user-attachments/assets/52c51188-3e63-4ac3-b9c2-7f93a6aebe24)


![Untitled 2](https://github.com/user-attachments/assets/6b202dbc-cda6-4749-b50e-8ca582aa4582)


- 미니 배치 사이즈 고르기 (hyperparameter): 속도를 빠르게 하기 위한 최적의 미니 배치 사이즈를 찾아내야 한다.
    - if mini batch size = m: 배치 경사 하강법 → 전체 훈련 데이터를 모두 훈련시키기 때문에, too long time per iteration (blue)
    - if mini batch size = 1: stochastic gradient descent (하나의 샘플이 하나의 batch가 된다) → 한 번에 하나의 샘플만 훈련시키기 때문에 (vectorization이 진행되지 않은 것과 마찬가지이기 때문에 속도 향상을 잃게 된다) 비효율적인 진행 방식이다. 또한 최적값 근처를 맴돌 뿐 절대 수렴하지 않는다(purple)
    - In practice: 1과 m 사이의 값을 사용한다. → 가장 좋음 (green)
        - 여러 개의 vectorization을 얻기 때문에 빠르다.
        - 전체를 여러 개의 batch으로 나눔, 전체 훈련 세트가 진행되기를 기다리지 않고 진행할 수 있다.
        
        <img width="639" alt="Untitled 3" src="https://github.com/user-attachments/assets/607b5f48-b6c3-4ef5-9e28-f958f5f6c5d7">
        
    - how to choose the size of mini batch?
        
        if train set ≤ 2000: 
        
        배치 경사 하강법을 사용 (전체 훈련 샘플을 하나의 batch로 설정)
        
        else:
        
        typical mini batch size: 64, 128, 256, 512 → 2의 제곱수를 사용하는 것을 추천
        
        -make sure mini batch fits in CPU / GPU memory: 정해진 CPU / GPU memory 크기 이내에서 memory를 사용하도록 mini batch size를 잘 정해야 한다. 
        

## 지수 가중 이동 평균

- 경사 하강법보다 빠른 최적화 알고리즘을 이해하기 위해선 지수 가중 이동 평균을 이해해야 한다.
- 지수 가중 이동 평균: 데이터의 이동 평균을 구할 때 과거의 데이터가 미치는 영향을 지수적으로 감쇄시켜 계산하는 방식으로, **최근 데이터에 더 높은 가중치를 둔다.** 최근 데이터에 더 많은 영향을 받는 데이터의 평균을 계산하기 위해 지수 가중 이동 평균을 구한다.
- 예를 들어 $\theta_{t}$를 t번째 날의 온도, $v_{t-1}$은 그 전 날의 온도 경향성이라고 했을 때, 지수 가중 이동 평균의 식은 다음과 같다. $v_{t}$는 t번째 날의 온도의 경향이다. 첫째 항은 과거의 경향성, 둘째 항은 새로운 경향성이다.
    
    $$
    ⁍
    $$
    
    -$v_{t}$ =  $\frac{1}{1-\beta}$동안의 기온의 평균
    
    -if $\beta$ = 0.9, then $v_{t}$ = 10일 동안 기온의 평균(red)
    
    → 0.98, 0.5보다 좋은 이동 평균을 제공한다. 
    
    -if $\beta$ = 0.98, then $v_{t}$ = 50일 동안 기온의 평균(green)
    
    → $\beta$ 값이 클수록 더 많은 날짜의 기온을 사용하기 때문에 곡선이 더 부드러워진다. 그러나 다양한 날짜의 기온을 사용하기 때문에, 이 곡선은 올바른 값에서 더 멀어진다. (곡선이 오른쪽으로 이동한다)
    
    -if $\beta$ = 0.5, then $v_{t}$ = 2일 동안 기온의 평균(yellow)
    
    → 데이터가 많지 않기 때문에 온도 변화에 더욱 민감하게 반응한다. 
    
    <img width="578" alt="Untitled 4" src="https://github.com/user-attachments/assets/a29216da-bbcc-408f-9dc3-d15ecbc204ce">
    

## 지수 가중 이동 평균 이해하기

![Untitled 5](https://github.com/user-attachments/assets/b6f14e81-bc06-41c8-b13a-3c7065f701f4)


- 과거의 $\theta$일수록 현재의 경향 $v_{t}$를 표현하는 데 더 적은 영향을 끼치고 있다. $\beta$가 0과 1 사이의 파라미터이기 때문이다.
- 위의 그래프는 $\theta$의 그래프, 아래의 그래프는 $\theta$의 계수 (0.1, 0.1 * 0.9….) 에 대한 그래프로, 지수적으로 감소하는 형태이다. $v_{t}$는 두 그래프를 element-wise하게 곱함으로써 구할 수 있다.

<img width="479" alt="Untitled 6" src="https://github.com/user-attachments/assets/3b2f11fb-065f-48fa-b655-6ea277588cf2">

- 얼마의 기간에 걸쳐 구한 평균인가?
    - $(1-\varepsilon)^{\frac{1}{\varepsilon}}$ = $\frac{1}{e}$
        - $1-\varepsilon = \beta$
        - $\frac{1}{\varepsilon}$은 몇 일에 걸친 평균 온도인지를 알 수 있다.
        - 예를 들어 $\varepsilon$이 0.1이라면 $\beta$ = 0.9이고 이는 10일에 걸친 온도의 평균이다.

```
v = 0
repeat{
	get next \theta_{t}
	# 지수 가중 이동 평균: 이동 평균을 계산하는 데 과거의 데이터가
		미치는 영향을 지수적으로 감쇄한다. 
	v_theta:= \beta*v_theta + (1-\beta)*\theta_{t}
}
```

- 지수 가중 이동 평균의 장점: 반복해도 v에 계속 값을 덮어씌우면 되기 때문에 메모리를 적게 사용할 수 있다. 또한 업데이트 식도 한 줄이라서 편리 & 효율적이다.

## 지수 가중 이동 평균의 편향보정

- bias correction

![c4754d79-704a-46dd-9364-0ed4fe2fcbb0](https://github.com/user-attachments/assets/9e8f5873-0c5e-4018-8cd2-3b0cd2c9149b)

- 만약 지수 가중 이동 평균의 식대로만 $v_{t}$를 구하면 (왼쪽 식) 실제로 그려지는 곡선은 보라색 곡선이 된다. 그러나 보라색 곡선에는 너무 낮은 값(편향)이 존재한다 → 편향을 제거해야 한다!
- 오른쪽 식처럼 $\theta$들의 가중 평균 $v_{t}$를 $1-\beta^t$로 나눠주면 $\theta$들의 가중 평균에서 편향이 제거된 초록색 그래프가 그려지고, 실제 값과 비슷하게 된다.
- t가 커질수록 $\beta^t$는 0에 가까워지므로($\because$ $\beta$는 0과 1사이의 파라미터) 편향 보정의 효과가 거의 없어지게 된다. → t가 커질수록 초록색과 보라색 그래프가 같아진다.
- 초기 단계에서 편향 보정으로 더 정확한 온도 평균의 추정값을 얻을 수 있다. (보라색 그래프에서 초록색 그래프로)

## Momentum 최적화 알고리즘

- 미니 배치 경사 하강법을 시행하면 다음과 같이 진동하며 최적점에 수렴한다. 그러나 이 방법은 경사 하강법의 속도를 느리게 하고 더 큰 learning rate를 사용하는 것을 막는다(오버슈팅 이슈). 또한 수직축에서는 진동을 막고 수평축에서는 왼쪽에서 오른쪽으로 학습이 빠르게 이루어지기를 원한다.

![Untitled 7](https://github.com/user-attachments/assets/f2e0d157-2c5a-4366-8ebe-04aa30321102)


- momentum 경사 하강법은 일반적인 경사 하강법보다 빠르게 작동한다.
    - 경사에 대한 지수 가중 평균 (VdW, Vdb)을 계산하고, 그 값으로 가중치를 업데이트한다.
    - 따라서 momentum 경사 하강법은 아래와 같은 알고리즘을 따른다.

```
	VdW = 0; Vdb = 0
	Iteration t:
		compute dW, db on current mini-batch
		V_dW = \beta*V_dW + (1 - \beta)*dW
		V_db = \beta*V_db + (1 - \beta)*db
		
		W:= W - \alpha*V_dW
		b:= b -\alpha*V_db
		
		# two hyperparameters: \alpha(learning rate), \beta
```

→ 이 알고리즘을 통해 진동을 줄여 훨씬 부드럽게 최적점에 수렴할 수 있게 된다. 수직축에서는 진동을 줄이면서도 수평축에서는 속도를 빠르게 한다.  이전 단계의 업데이트 방향을 일부 기억함(관성)으로써 매개변수 업데이트 시 이전 단계의 기울기를 일정 부분 반영함으로써 알고리즘의 수렴을 가속화하고 진동을 줄여준다. 베타가 클수록 이전 단계 경로의 방향성을 강하게 유지시키므로 진동의 폭이 좁아지게 된다. 

![Untitled 8](https://github.com/user-attachments/assets/2438c5ed-c444-4dd2-8e62-c893228777e4)


→ 또한, 밥그릇 모양에서 공을 굴려 최적점으로 보낼 때, $\beta$는 1보다 작은 값이므로 friction, VdW, Vdb는 공의 velocity, dW, db는 공의 momentum(acceleration)의 역할을 한다고 이해할 수 있다. 

→ momentum 경사 하강법을 구현할 때는 편향 보정을 거의 하지 않는다. step이 10번 이상이면 이동 평균에 더 이상 편향 보정을 하는 효과가 미미하기 때문이다. 

→ 몇몇 논문에서 VdW를 계산할 때 $1-\beta$를 생략하는데, 이는 VdW를 1/$(1-\beta)$로 scaling한 것이다. 따라서 $\alpha = \frac{1}{(1-\beta)}$가 되어야 한다. 그러나 실제로는 두 가지 방법 모두 잘 작동한다. (교수님은 1-$\beta$가 있는 수식을 더 선호한다)

## RMSProp 최적화 알고리즘

- root mean square prop algorithm → 마찬가지로 경사 하강법을 빠르게 한다. 과거의 기울기는 적게 반영하고 최근의 기울기는 많이 반영하는 지수 가중 평균을 사용한다.
    - 수평 방향(W): 빠르게 이루어지기를 원함/ 수직 방향(b): 느리게 이루어지기를 원함(진동 적게)
    
    ```
    Iteration t:
    	compute dW, db on current mini-batch
    		# 도함수의 제곱을 지수 가중 평균한 것
    		# dW^2은 element-wise하게 제곱한 
    		S_dw = \beta*S_dw + (1 - \beta)*dW^2 <- small
    		S_db = \beta*S_db + (1 - \beta)*db^2 <- large
    		
    		W:= W - \alpha*dW/\root{S_dW}
    		b:= b - \alpha*db/\root{S_db}
    	
    ```
    
    - 수평 방향(W)에서의 기울기는 완만해 미분값 SdW의 값은 상대적으로 작다. 따라서 수평 방향(W)은 기존의 learning rate보다 빠르게 업데이트되고, 이는 더 빠르게 수렴하도록 한다. 반면 수직 방향(b)에서의 기울기가 훨씬 가파르기 때문에 미분값 Sdb가 상대적으로 7크다. 따라서 수직 방향(b)으로는 기존의 learning rate보다 느리게 업데이트되고, 이는 진동을 줄인다.
    - 이 알고리즘을 사용하면 진동을 줄이면서도 빠르게 최적점으로 수렴할 수 있다.
    
    ![Untitled 9](https://github.com/user-attachments/assets/16f8d2a0-ddd2-457d-af24-67863161da74)

    
    - 주의할 점: W, b 업데이트 식에서 SdW, Sdb의 값이 너무 작지 않도록 한다(폭발할 수 있음) → SdW, Sdb에 $\varepsilon$ = 10^(-8)을 더함으로써 보완 가능

## Adam 최적화 알고리즘

- Adam 최적화 알고리즘 = momentum + RMSProp
- Adam = Adaptive moment estimation

```
# 초기화
VdW = 0; SdW = 0; Vdb = 0; Sdb = 0

Iteration t:
	compute dW, db using current mini-batch
	
	# momentum
	V_dW = B1*V_dW + (1 - B1)*dW
	V_db = B1*V_db + (1 - B1)*db
	
	# RMSProp
	S_dW = B2*S_dW + (1 - B2)*dW^2
	S_db = B2*S_db + (1 - B2)*db^2
	
	# 편향 보정 (일반적으로 Adam에서는 편향 보정을 한다.)
	V_dW_corrected = V_dW / (1 - B1^t)
	V_db_corrected = V_db / (1 - B1^t)
	S_dW_corrected = S_dW / (1 - B2^t)
	S_db_corrected = S_db / (1 - B2^t)
	
	# W, b 업데이트
	# 분모가 0이 되는 것을 막기 위해 분모에 엡실론을 더해줌
	W:= W-A*V_dW_corrected / \root(S_dW_corrected) + E
	b:= b-A*V_db_corrected / \root(S_db_corrected) + E
```

- 여러 개의 hyperparameter가 있다 ($\alpha$ 빼고 대부분 보정하지 않는다)
    - $\alpha$: learning rate → needs to be tuned
    - $\beta_1$: 0.9 (dW)
    - $\beta_{2}$: 0.999 (dW^2)
    - $\varepsilon$: 10^(-8)

## 학습율 감쇠

- 학습율 감쇠: 시간에 따라 learning rate를 감소시키는 것 → 학습 알고리즘의 속도를 높이는 한 가지 방법
- 고정된 학습율 사용: 노이즈 발생, 최적점에 수렴 x
- 시간에 따라 학습율이 줄어들면 step size가 작아짐에 따라 최적점에 더 잘 수렴할 수 있다.
- 1 epoch = 1 pass through data
    - $\alpha = \frac{1}{1+decayrate*epochnum}*\alpha_0$: hyperparameter $\alpha_0,$ decay_rate 조절
    - exponential decay: $\alpha = 0.95^{epochnum}*\alpha_0$
    - $\alpha = \frac{k}{epochnum}*\alpha_0$
    - $\alpha = \frac{k}{t**0.5}*\alpha_0$: t는 미니 배치 개수
    - 이산 감쇠: 계단처럼 이산적으로 학습율 감소

## 지역 최적값 문제

- 고차원 비용함수에서 경사가 0인 경우는 대부분 지역 최적값이 아니라 안장점이다.
    - 안장점 주위는 대부분 경사가 0이므로 지역 최적값에 갇힐 위험이 있다 → 그러나 big neural network를 사용하면 괜찮다.
    - 지역 최적값에 빠지면 대부분 경사가 0에 가깝기 때문에 학습이 느려질 수 있다. 또한, 다른 방향으로의 전환이 없다면 안장 지대에서 벗어나기 힘들다 → Adam, RMSProp과 같은 최적화 알고리즘을 통해 빠져나올 수 있다.

![Untitled 10](https://github.com/user-attachments/assets/2147a6c6-9e8c-4160-856b-081dfdeef60f)


## Quiz

![Untitled 11](https://github.com/user-attachments/assets/fdc9a857-8e4c-43c9-8073-87d95ced7d61)


- 만약 미니배치의 크기가 1이라면, 미니배치 내의 훈련 샘플들 간의 벡터화의 장점을 상실한다.
- 만약 미니배치의 크기가 m이라면, 한 번의 진전을 이루기 전에 (한 번의 epoch이 끝나기 전에) 모든 훈련 샘플들을 처리해야 하는 배치 경사 하강법으로 빠지게 된다. → 한 번의 epoch마다 너무 오래 시간이 걸린다)

## 캐글 필사

- `torch.tensor`: dtype 지정 가능
- `torch.Tensor`: 실수형 텐서 반환, 직접 텐서 값을 정할 때
- Variable은 더 이상 사용 x, requires_grad = True를 통해 자동미분 기능 지원
- 모델 사용
    - 모델 클래스 생성
    - 모델 인스턴스 생성
    - criterion
    - optimizer
    - Train
        - optimizer.zero_grad()
        - outputs = model(inputs)
        - loss = criterion(outputs, targets)
        - loss.backward()
        - optimizer.step()
        - current_loss += loss.item()
    - Test data 예측
