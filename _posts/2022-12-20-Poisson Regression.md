---
layout: post
title:  "Poisson Regression"
date: 2022-12-20
author: seolbluewings
categories: Statistics
---

[작성중...]

종속변수($$y$$)가 음수가 될 수 없는 정수값을 가지며 셀 수 있는(countable) 데이터이면서 1달 간의 로그인 횟수, 마케팅 반응횟수 같이 어떠한 정해진 시간동안의 event 발생횟수를 의미하는 값일 때, 일반화 선형모형(Generalized Linear Model)의 한 종류인 포아송 회귀모형(Poisson Regression)을 활용할 수 있다.

$$y_{i}$$가 특정 시간동안의 event 발생횟수일 때, $$y_{i}$$는 평균이 $$\mu_{i}$$인 포아송 분포를 따른다고 할 수 있다.

$$
\begin{align}
y_{i} &\sim \text{Poi}(\mu_{i}) \nonumber \\
f(y_{i}\vert \mu_{i}) &= \text{exp}\left(y_{i}\log{\mu_{i}}-\mu_{i}-\log{(y_{i}!)}  \right) \nonumber
\end{align}
$$

또한 데이터가 $$ \mathbf{x_{i}} = (1,x_{i1},x_{i2},...,x_{ip}) $$ 와 같은 형태로 주어졌을 때, $$\mu_{i}$$ 에 대한 link function $$g(\mu_{i})$$ 을 활용하여 회귀식을 fitting 한다.

$$ g(\mu_{i}) = \mathbf{x}^{T}\beta $$

일반적으로 link function $$g(\mu_{i})$$로 log함수를 활용한다. 포아송 회귀의 경우, 회귀계수 $$\beta$$는 Fisher Scoring이란 방식을 통해 계산을 진행하게 되는 것으로 알려져 있는데 link function을 log함수로 설정하면 Fisher Scoring은 Newton's Method와 동일해지는 것으로 알려져 있다. 자세한 회귀계수 $$\beta$$에 대한 유도는 이 [링크](https://zephyrus1111.tistory.com/68)에서 확인할 수 있다.

대다수 데이터가 발생횟수 0에 몰려있고 0 아닌 관측치가 일부 존재하는 경우에는 Zero-Inflated Poisson Regression 라는 방식을 적용한다. 이 Zero-Inflated Poisson Regression의 경우, $$\pi$$의 확률로 0의 값을 가지며, $$1-\pi$$의 확률로 0 아닌 값을 갖는다.

$$
\begin{align}
p(y_{i}=0) = \pi + (1-\pi)\text{exp}(-\mu_{i}) \nonumber \\
p(y_{i}\geq 1) = (1-\pi)\frac{\mu^{y_{i}}\text{exp}(-\mu_{i})}{y_{i}!} \nonumber
\end{align}
$$

일반적인 Poisson Regression과 달리 2개의 parameter를 갖는 모델이라 할 수 있다. 



#### 참조 문헌
1. [Count 데이터-Poisson Log Linear Model 적합하기 with Python](https://zephyrus1111.tistory.com/88) <br>

