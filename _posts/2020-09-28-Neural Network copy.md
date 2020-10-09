---
layout: post
title:  "신경망(Neural Network) 개요"
date: 2020-09-28
author: seolbluewings
categories: Statistics
---

[작성중...]

신경망(Neural Network) 모델은 딥러닝을 이해하기 위해서 가장 기본이 되는 개념이라 할 수 있다. 우리 몸속의 뉴런과 뉴런이 연결되어 서로 신호를 주고 받는 것처럼 Neural Network는 아래의 그림과 같이 입력층(Input layer)의 값 $$\mathbf{X} = (x_{1},...,x_{n})$$ 을 받아서 이를 은닉층(hidden layer)로 전달하게 되고 정해진 연산을 수행하여 발생한 결과를 또 다음에 존재하는 hidden layer 또는 출력층(output layer)로 전송하게 된다. output layer에서도 연산을 거쳐 최종 결과를 도출해낸다.

![NN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/NN.png?raw=true){:width="70%" height="70%"}{: .center}

이 그림처럼 다수의 input을 전달하여 하나의 output을 도출해내는 것을 퍼셉트론(Perceptron)이라 부른다. input variable은 보통 numerical variable로 구성되며 만약 categorical 유형이라면 이를 dummy화하여 input으로 설정하여야 한다.

이렇게 설정한 입력값(input variable)은 가중치(weight)를 적용하여 선형 결합(linear combination)을 형성한다. $$\mathbf{w} = (w_{1},...,w_{n})$$ 이라면 다음 뉴런(층)으로 전달되는 값은 $$\mathbf{w}^{T}\mathbf{X} = \sum_{i=1}^{n}w_{i}x_{i}$$ 이다. 선형결합 결과를 전달받은 뉴런은 해당층의 임계값과 선형결합 결과를 비교하고 활성화 함수(activation function)을 이용하여 값을 출력해낸다.

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


#### 오차 역전파(Error BackPropagation)

입력층 방향에서 출력층 방향으로 값이 전달되는 것을 순전파라고 부르는데 이 반대 방향으로 이동하는 것을 역전파라고 한다. 앞서 우리는 단층(1-hidden layer) 형태의 NN을 살펴보았는데 일반적으로는 학습능력이 더 뛰어난 다층 신경망 모델을 활용한다. 이 다층 신경망 모델을 학습하는 과정에서 오차 역전파(Error Backpropagation) 기법이 활용된다.

먼저 훈련 데이터 $$\mathcal{D} = \{(x_{1},y_{1}),...(x_{m},y_{m})\}$$ 이 존재한다고 가정하자. $$x_{1} = (x_{11},x_{12},...x_{1d})$$ 이고 $$y_{1} = (y_{11},...,y_{1l})$$ 이라 하자. 즉, 입력층은 뉴런 d개, 출력층은 l개이며 중간에 존재하는 은닉층은 q개라고 하자. 훈련 데이터 중 1개의 sample값을 가지고 표준 오차 역전파법에 대해서 살펴보자.

아래의 그림과 같이 입력층, 은닉층, 출력층이 있다고 할 때, 은닉층 h번째 뉴런의 입력값과 출력층 j번째 뉴런의 입력값은 다음과 같이 정의될 수 있다. 그리고 2개 층 모두 시그모이드 함수를 사용한다고 가정하자.

$$
\begin{align}
	\alpha_{hk} &= \sum_{i=1}^{d}v_{ih}x_{ik} \nonumber \\
    \beta_{jk} &= \sum_{h=1}^{q}w_{hj}b_{hk} \nonumber
\end{align}
$$

훈련 데이터들 중 $$(x_{k},y_{k})$$ sample에 대하여 신경망 출력층 값은 $$\hat_{y_{k}} = (\hat{y_{1k}},...,\hat{y_{lk}})$$ 로 표현하기로 하자. 시그모이드 함수를 $$f(x)$$라 표현한다고 했을 때, $$\hat{y_{jk}} = f(\beta_{jk}-\theta_{jk}) $$ 라 할 수 있다. 여기서 $$\theta$$는 출력층에서 활용되는 임계값에 대한 parameter이다. 그리고 최종적으로 우리에게 필요한 평균 오차값은 $$E_{k} = \frac{j=1}{l}(y_{jk}-\hat{y_{jk}})^{2}$$ 일 것이다. 


![NN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/NN_EXAMPLE.png?raw=true){:width="70%" height="70%"}{: .center} 



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)

3. [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)