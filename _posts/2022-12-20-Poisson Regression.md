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

또한 데이터가 $$ \matbhf{x_{i}} = (1,x_{i1},x_{i2},...,x_{ip}) $$ 와 같은 형태로 주어졌을 때, $$\mu_{i}$$ 에 대한 link function $$g(\dot)$$ 을 활용하여 회귀식을 fitting 한다.

$$ g(\mu_{i}) = \mathbf{x}^{T}\beta $$

일반적으로 





#### 참조 문헌
1. [Count 데이터-Poisson Log Linear Model 적합하기 with Python](https://zephyrus1111.tistory.com/88) <br>

