---
layout: post
title:  "신경망(Neural Network) 개요"
date: 2020-09-28
author: seolbluewings
categories: Statistics
---

신경망(Neural Network) 모델은 딥러닝을 이해하기 위해서 가장 기본이 되는 개념이라 할 수 있다. 우리 몸속의 뉴런과 뉴런이 연결되어 서로 신호를 주고 받는 것처럼 Neural Network는 아래의 그림과 같이 입력층(Input layer)의 값 $$\mathbf{X} = (x_{1},...,x_{n})$$ 을 받아서 이를 은닉층(hidden layer)로 전달하게 되고 정해진 연산을 수행하여 발생한 결과를 또 다음에 존재하는 hidden layer 또는 출력층(output layer)로 전송하게 된다. 은닉층은 명백하게 값이 존재하는 입력층, 출력층과 달리 실제 우리 눈에는 보이지 않는 단계라 할 수 있다. 출력층에서는 마찬가지로 연산을 거쳐 최종 결과를 도출해낸다.

![NN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/NN.png?raw=true){:width="70%" height="70%"}{: .center}

이 그림처럼 다수의 input을 전달하여 하나의 output을 도출해내는 것을 퍼셉트론(Perceptron)이라 부른다. input variable은 보통 numerical variable로 구성되며 만약 categorical 유형이라면 이를 dummy화하여 input으로 설정하여야 한다.

이렇게 설정한 입력값(input variable)은 가중치(weight)를 적용하여 선형 결합(linear combination)을 형성한다. $$\mathbf{w} = (w_{1},...,w_{n})$$ 이라면 다음 뉴런(층)으로 전달되는 값은 $$\mathbf{w}^{T}\mathbf{X} = \sum_{i=1}^{n}w_{i}x_{i}$$ 이다. 선형결합 결과를 전달받은 뉴런은 해당층의 임계값과 선형결합 결과를 비교하고 입력값을 출력값으로 변환시키는 활성화 함수(activation function)을 이용하여 값을 출력해낸다.

활성화 함수는 개별 뉴런으로 전달되는 입력값을 출력값으로 변환하는 함수이며 일반적으로 비선형 함수를 활용한다. 비선형 함수를 활용하는 이유는 다음과 같다. 이 이유는 [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)에 적절한 예시가 있어 해당구절을 가져와 본다.

> 활성화 함수로 선형함수를 이용하면 은닉층을 만드는 이유가 사실상 없다.만약 선형함수 $$h(x)=cx$$ 를 활성화 함수로 사용하고 은닉층을 2개로 한다면, 출력층 결과는 $$y(x)=h(h(x))=c^{2}x$$ 으로 결정되는데 이는 $$a=c^{2}$$ 이라 했을 때 $$h(x)=ax$$와 사실상 다를게 없기 때문이다. 선형함수로는 여러 은닉층을 구성하는 이점을 살릴 수 없다.

가장 빈번하게 사용되는 활성화 함수는 시그모이드 함수 $$\text{sigmoid}(x) = \frac{1}{1+e^{-x}}$$ 이며 step function 또는 hyperbolic tangent function, ReLU를 사용하기도 한다. 앞으로는 시그모이드 함수를 사용하는 것으로 한정지어 이야기를 풀어갈 것이다.

#### 간단한 단층 신경망으로 익숙해지기

![NN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/NN1.png?raw=true){:width="70%" height="70%"}{: .center}

먼저 가장 간단한 NN 형태를 살펴보자. 다음과 같은 Neural Network가 있다고 가정하자. $$\mathbf{X}=(x_{1},x_{2})$$가 있고 hidden layer는 1개층, output layer는 1개 node로 $$\hat{y}$$를 출력한다고 하자.

참고로 Neural Network 모델은 각 node마다 fully connected 되어있으며 같은 층의 node끼리는 연결되지 않으며 또 층을 뛰어넘는 연결도 존재하지 않는다. 지금의 경우는 1개의 hidden layer층을 갖는 network이나 hidden layer를 2개 이상 설정할 수 있다. 이렇게 설정한다면, 이를 multi-layer perceptron(MLP) 모델이라 부르기도 한다.

각 가중치와 입력값을 활용하여 생성한 선형결합 결과$$b_{1},b_{2},b_{3}$$는 다음과 같을 것이다.

$$
\begin{align}
b_{1} &= w_{1}x_{1}+w_{2}x_{2} \nonumber \\
b_{2} &= w_{3}x_{1}+w_{4}x_{2} \nonumber \\
b_{3} &= w_{5}x_{1}+w_{6}x_{2} \nonumber
\end{align}
$$

이 $$\mathbf{b}=(b_{1},b_{2},b_{3})$$ 는 각 node에서의 활성화 함수 결과를 해당 node의 최종결과값으로 갖는다.

$$
\begin{align}
n_{1} &= \frac{1}{1+e^{-b_{1}}} \nonumber \\
n_{2} &= \frac{1}{1+e^{-b_{2}}} \nonumber \\
n_{3} &= \frac{1}{1+e^{-b_{3}}} \nonumber
\end{align}
$$

출력층으로 전달되는 입력값은 $$b_{4} = w_{7}n_{1}+w_{8}n_{2}+w_{9}n_{3}$$ 이며 최종적 결과는 $$\hat{y} = \frac{1}{1+e^{-b_{4}}}$$ 를 통해 산출해낼 수 있다.

#### 출력층 설계

신경망 문제는 분류와 회귀 문제에 모두 사용될 수 있다. 일반적으로 회귀함수의 경우 활성화 함수를 항등함수를 사용하고 분류문제의 경우는 소프트맥스(softmax) 함수를 사용한다.

분류 문제에서 사용하는 소프트맥스 함수의 형태는 다음과 같다.

$$
y_{k} = \frac{\text{exp}(a_{k})}{\sum_{i=1}^{n}\text{exp}(a_{i})}
$$

이 결과는 $$[0,1]$$ 범위에 존재하는 실수이며, 모든 출력의 총합이 1이므로 사실상의 확률과도 같다고 볼 수 있다. 신경망을 이용해 분류를 진행할 때, 우리는 일반적으로 가장 큰 출력값을 보이는 뉴런에 해당하는 클래스로 인식하는 것으로 생각한다.

출력층 뉴런의 개수는 문제에 specific하게 설정한다. 일반적으로 분류하고자 하는 class 개수로 설정하게 되는데 예를 들어, MNIST 같이 0부터 9까지의 손글씨를 인식하는 문제의 경우 $$y_{0}=0, y_{1}=1,...,y_{9}=9$$ 과 같이 10개의 출력층 뉴런을 설정한다.


#### 오차 역전파(Error BackPropagation)

입력층 방향에서 출력층 방향으로 값이 전달되는 것을 순전파라고 부르는데 이 반대 방향으로 이동하는 것을 역전파라고 한다. 앞서 우리는 단층(1-hidden layer) 형태의 NN을 살펴보았는데 일반적으로는 학습능력이 더 뛰어난 다층 신경망 모델을 활용한다. 이 다층 신경망 모델을 학습하는 과정에서 오차 역전파(Error Backpropagation) 기법이 활용된다. 

먼저 훈련 데이터 $$\mathcal{D} = \{(x_{1},y_{1}),...(x_{m},y_{m})\}$$ 이 존재한다고 가정하자. $$x_{1} = (x_{11},x_{ik},...x_{dk})$$ 이고 $$y_{1} = (y_{1k},...,y_{lk})$$ 이라 하자. 즉, 입력층은 뉴런 d개, 출력층은 l개이며 중간에 존재하는 은닉층은 q개라고 하자. 훈련 데이터 중 1개의 sample값을 가지고 표준 오차 역전파법에 대해서 살펴보자.

아래의 그림과 같이 입력층, 은닉층, 출력층이 있다고 할 때, 은닉층 h번째 뉴런의 입력값과 출력층 j번째 뉴런의 입력값은 다음과 같이 정의될 수 있다. 그리고 2개 층 모두 시그모이드 함수를 사용한다고 가정하자.

$$
\begin{align}
	\alpha_{hk} &= \sum_{i=1}^{d}v_{ih}x_{ik} \nonumber \\
    \beta_{jk} &= \sum_{h=1}^{q}w_{hj}b_{hk} \nonumber
\end{align}
$$

훈련 데이터들 중 $$(x_{k},y_{k})$$ sample에 대하여 신경망 출력층 값은 $$ \hat{y}_{k} = (\hat{y}_{1k},...,\hat{y}_{lk})$$ 로 표현하기로 하자. 시그모이드 함수를 $$f(x)$$라 표현한다고 했을 때, $$\hat{y}_{jk} = f(\beta_{jk}-\theta_{jk}) $$ 라 할 수 있다. 여기서 $$\theta$$는 출력층에서 활용되는 임계값에 대한 parameter이다. 그리고 최종적으로 우리에게 필요한 평균 오차값은 $$E_{k} = \sum_{j=1}^{l}(y_{jk}-\hat{y_{jk}})^{2}$$ 일 것이다.

![NN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/NN_EXAMPLE.png?raw=true){:width="40%" height="30%"}{: .center}

우리가 구해야하는 parameter의 개수는 다음과 같다. 먼저 입력층에서 은닉층으로 향하는 $$d \times q$$개, 은닉층에서 출력층으로 가는 $$ q \times l $$개, 은닉층의 임계값 $$q$$개, 출력층의 임계값 $$l$$개로 총 $$(d+l+1)q+l$$ 개의 parameter를 정해야 한다. 오차 역전파 알고리즘은 재귀적 학습을 통해 반복 과정에서 퍼셉트론의 학습규칙으로 parameter에 대한 예측값을 업데이트하는 방식을 취한다. 즉, 임의의 parameter $$v$$에 대하여 다음과 같이 값이 업데이트 된다.

$$ v \leftarrow v + \Delta v $$

parameter에 대한 Loss Function의 미분은 "parameter 값을 변화시켰을 때, Loss Function이 어떻게 변화하는가?" 에 대한 답을 준다. 만약 이 미분값이 음수라면 parameter를 양의 방향으로 업데이트 시키며, 미분값이 양수라면 parameter를 음의 방향으로 업데이트 시킨다. 즉, Gradient Descent 방식을 취하는 것이다.

은닉층에서 출력층으로 값을 전달할 때의 parameter $$w_{hj}$$를 예시로 값을 어떻게 업데이트 시키는지 알아보자. parameter $$w_{hj}$$의 Gradient Descent값은 다음과 같이 표현된다.

$$
\Delta w_{hj} = -\eta\frac{\partial E_{k}}{\partial w_{hj}} \quad \eta \in (0,1)
$$

parameter $$w_{hj}$$가 어떻게 오차 $$E_{k}$$에 영향을 미치는지 생각해보자. 우선 $$w_{hj}$$는 j번째 출력층 뉴런의 입력값 $$\beta_{jk}$$에 영향을 미친다. 여기서 생성된 $$\beta_{jk}$$는 $$\hat{y}_{jk}$$ 에 영향을 미칠 것이며 궁극적으로 이 $$\hat{y}_{jk}$$가 $$E_{k}$$ 값을 결정짓게 된다. 따라서 Chain Rule을 이용하여 다음과 같이 표현할 수 있을 것이다.

$$
\frac{\partial E_{k}}{\partial w_{hj}} = \frac{\partial E_{k}}{\partial \hat{y}_{jk}} \times \frac{\partial \hat{y}_{jk}}{\partial \beta_{jk}} \times \frac{\beta_{jk}}{w_{hj}}
$$

그리고 우리는 앞서 $$\beta_{jk}$$를 $$\beta_{jk} = \sum_{h=1}^{q}w_{hj}b_{hk}$$ 로 정의하였다. 따라서 $$\frac{\partial \beta_{jk}}{\partial w_{hj}} = b_{hk}$$ 라고 표현할 수 있다.

시그모이드 함수는 함수의 특성상 $$\Delta f(x) = f(x)(1-f(x))$$ 로 표현된다.

$$
\begin{align}
-\frac{\partial E_{k}}{\partial \hat{y}_{jk}} \times \frac{\partial \hat{y}_{jk}}{\partial \beta_{jk}} &= -\Delta f(\beta_{jk}-\theta_{jk}) \times (\hat{y}_{jk}-y_{jk}) \nonumber \\
&= \hat{y}_{jk}(1-\hat{y}_{jk})(\hat{y}_{jk}-y_{jk}) \nonumber
\end{align}
$$

결국 $$\Delta w_{hj} = \eta\hat{y}_{jk}(1-\hat{y}_{jk})(\hat{y}_{jk}-y_{jk})b_{hk}$$ 로 표현된다.

비슷한 방법으로 다음과 같이 Gradient 값을 구할 수 있다. 여기서 $$\theta_{jk},\gamma_{hk}$$ 는 각각 출력층, 은닉층에서의 임계값 parameter를 의미한다.
 
$$
\begin{align}

\Delta\theta_{jk} &= -\eta\hat{y}_{jk}(1-\hat{y}_{jk})(\hat{y}_{jk}-y_{jk}) \nonumber \\
\Delta v_{ih} &= \eta \left(b_{hk}(1-b_{hk})\sum_{j=1}^{l}w_{hj}\hat{y}_{jk}(1-\hat{y}_{jk})(\hat{y}_{jk}-y_{jk})\right)x_{ik} \nonumber \\
\Delta\gamma_{hk} &= -\eta\left(b_{hk}(1-b_{hk})\sum_{j=1}^{l}w_{hj}\hat{y}_{jk}(1-\hat{y}_{jk})(\hat{y}_{jk}-y_{jk})\right) \nonumber
\end{align}
$$

즉, 오차 역전파 알고리즘은 먼저 입력 데이터를 활용하여 순방향으로 신경망 모델을 작동시키고 출력층에서 결과값을 얻어내고서 출력층의 오차를 계산하고 해당 오차를 역전파하여 은닉층 뉴런으로 전달한다. 여기서 오차에 다라 가중치와 임계값을 조정하고 오차가 극히 작아질 때까지 이를 반복 수행한다.

위에서 우리는 하나의 데이터 sample $$(x_{k},y_{k})$$ 에 대하여 역전파 알고리즘이 작동하는 방식을 살펴보았는데 훈련 데이터 $$\mathcal{D}$$의 사이즈가 클 경우, 모든 데이터셋을 활용해 한번에 역전파 알고리즘을 진행시키는 것보다 개별 데이터로 역전파 알고리즘을 적용시키는 것이 더 좋은 결과를 가져오는 것으로 알려져있다.

#### 왜 Loss Function을 활용하는가?

앞선 소개에서는 훈련 데이터를 통해 parameter의 최적값을 얻어낼 때, 오차제곱합이라는 손실 함수(Loss Function)을 활용했다. 만약 신경망을 통해 해결하고자 하는 문제가 분류 문제라면, 정확도(Accuracy)라는 지표가 있을 것이다. 왜 정확도가 아닌 Loss Function을 이용해서 parameter를 업데이트 시켜나가는가?

그 이유는 parameter를 미세하게 조정할 경우, 정확도가 개선되지 않고 일정하게 유지되기 때문이다. 우리는 미분을 통해 parameter 값을 업데이트 시켜나갔는데 정확도를 활용할 경우 미분값이 0이 나와 업데이트 되지 않는 경우가 빈번하게 발생할 수 있다.

만약 100개의 데이터 중 32개를 제대로 분류한다고 하면, 32%라고 정확도를 말할 수 있다. 여기서 parameter를 변화시킨다고 했을 때, 여전히 분류 정확도가 32%일 수 있으며 변한다 할지라도 32.5% 또는 33%로 불연속적인 값으로 변화한다. 반면, Loss Function을 이용할 경우 연속적인 값을 얻어낼 수 있다.

불연속성이 갖는 문제는 활성화 함수로 계단 함수(Step function)을 잘 이용하지 않는 이유이기도 하다. 우리는 시그모이드 함수를 주로 활용하게 되는데, 이는 시그모이드 함수는 미분값으로 0을 갖지 않기 때문이다.



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)

3. [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)