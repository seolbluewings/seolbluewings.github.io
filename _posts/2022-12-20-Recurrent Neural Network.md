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








#### 참조 문헌
1. [Count 데이터-Poisson Log Linear Model 적합하기 with Python](https://zephyrus1111.tistory.com/88) <br>

