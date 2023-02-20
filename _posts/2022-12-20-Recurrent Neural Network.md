---
layout: post
title:  "Recurrent Neural Network"
date: 2022-12-20
author: seolbluewings
categories: Statistics
---

[작성중...]

인공 신경망 모델 중에서 RNN(Recurrent Neural Network) 모델은 순서가 있는 데이터에 대한 예측을 목적으로 시계열 데이터, 자연어 처리 등을 위해 사용되고 있다.

RNN 명칭에서 Recurrent('순환')라는 표현에 주목할 필요가 있다. 어느 한 지점에서 시작한 것이 일정 시간의 흐름 뒤 다시 원래의 장소로 돌아오는 것을 Recurrent라고 표현할 수 있고 RNN은 이러한 순환 경로 형태가 담긴 모델이다.

이 순환 경로를 따라서 데이터는 끊임없이 순환할 수 있고 데이터가 순환하기 때문에 RNN 모델은 과거의 데이터를 기억하는 동시에 최신 데이터를 반영해 갱신될 수 있다.

RNN 모델을 도식화한 이미지는 아래와 같으며 이미지의 왼쪽과 같이 순환하는 경로를 확인할 수 있다.

![RNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/rnn_1.png?raw=true){:width="90%" height="80%"}{: .aligncenter}

시계열 데이터 $$(x_{0},x_{1},...,x_{t},...)$$를 RNN의 input으로 입력되고 은닉층(h)의 데이터 $$(h_{0},h_{1},...,h_{t},...)$$가 생성된다. 은닉층을 통해서 output layer가 만들어지는데 RNN에서 주목해서 확인할 사항은 바로 은닉층에서 나와 다른 은닉층으로 전달되는 화살표이다. 특정 시점 t에서의 은닉층 벡터가 다음 시점 t+1의 입력 벡터로 역할하는 것을 확인할 수 있다.

이 그림에서 주의해야할 점은 다수의 RNN 계층이 모두 사실은 '같은 level의 계층'이라는 것이다. 시각이 다를 뿐 같은 계층이다.

이처럼 각 시각의 RNN 계층은 해당 계층으로의 입력값과 1시점 전의 RNN 계층의 출력값을 input으로 삼아 현 시점의 출력을 계산한다.

$$ \mathbf{h}_{t} = \text{tanh}(\mathbf{h}_{t-1}\mathbf{W}_{h} + \mathbf{x}_{t}\mathbf{W}_{x}+\mathbf{b}) $$

현 시점의 결과값 $$\mathbf{h}_{t}$$는 다음 계층을 향해 출력되는 동시에 1시점 뒤의 RNN 계층(자기자신)을 향해서도 출력 된다. 이처럼 하나의 은닉층에서의 값이 1시점 뒤에 다시 갱신되어 RNN 계층을 '메모리를 가진 계층' 으로 표현하기도 한다.

RNN도 신경망 문제로 weight parameter를 정확하게 계산하기 위해 Backpropagation을 수행하는데 시간까지 고려한 Backpropagation을 수행해야 한다. 다만 모든 시점의 데이터를 Backpropagation 하려면 리소스 이슈가 있기 때문에 신경망을 적당한 길이에서 끊어버린다. 단 순전파는 절단하지 않고 역전파만 적정선에서 끊어버리는 것이다.

순전파는 계속 이어지게 구성하고 역전파는 적정선에서 절단하는 것은 2가지 의미가 있다. 하나는 순전파가 지속적으로 이어지기 때문에 RNN에서는 데이터를 순서대로(sequential) 입력해야한다는 것이며 역전파의 연결을 잘라버리기 때문에 분석 과정에서 지나친 먼 미래의 데이터에 대해서는 고려하지 않아도 된다. Backpropagation에 대해서는 다음의 [링크](https://towardsdatascience.com/backpropagation-in-rnn-explained-bdf853b4e1c2)에서 수식 등 참고하면 좋을 것이다.











#### 참조 문헌
1. [Count 데이터-Poisson Log Linear Model 적합하기 with Python](https://zephyrus1111.tistory.com/88) <br>

