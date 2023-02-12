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

![RNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/rnn_1.png?raw=true){:width="80%" height="80%"}{: .aligncenter}




#### 참조 문헌
1. [Count 데이터-Poisson Log Linear Model 적합하기 with Python](https://zephyrus1111.tistory.com/88) <br>

