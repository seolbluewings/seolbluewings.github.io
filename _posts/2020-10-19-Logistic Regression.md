---
layout: post
title:  "로지스틱 회귀(Logistic Regression)"
date: 2020-10-19
author: seolbluewings
categories: 선형모델
---

[작성중...]


로지스틱 회귀는 프로빗 모델처럼 반응변수 범주형인 케이스에서 활용할 수 있는 방법이다. 로지스틱 회귀는 새로운 변수(X)가 주어졌을 때, 반응 변수가 각 범주에 속할 확률이 얼마인지를 추정하며, 추정된 확률에 따라 반응 변수의 Class를 분류하게 된다. 로지스틱 회귀 역시도 프로빗 회귀처럼 일반화 선형 모델(generalized linear model)의 한 종류라고 할 수 있다.

일단 반응 변수의 클래스가 2가지인 경우 ,$$ y \in \{0,1\}$$ 로 표현되는 경우를 생각해보자.

앞서 일반화 선형 모델은 연결함수(link function) $$h$$를 이용해서 실수값 $$\mathbf{X}\beta$$를 0 또는 1이란 값으로 변경시킨다.

이런 상황에서 연결함수로 시그모이드 함수(로지스틱 함수)를 활용하는 경우가 있는데 이를 로지스틱 회귀 모델이라 부르며 다음과 같이 표현할 수 있다.

$$
\begin{align}
\pi(\mathbf{X}) &= p(y=1\mid\mathbf{X}) \nonumber \\
\log{\frac{\pi(\mathbf{X})}{1-\pi(\mathbf{X})}} &= \mathbf{X}\beta \nonumber \\
\pi(\mathbf{X}) = \frac{e^{\mathbf{X}\beta}}{1+e^{\mathbf{X}\beta}} \nonumber
\end{align}
$$

여기서 로지스틱 회귀의 중요한 특징을 하나 발견할 수 있다. 수식의 형태를 고려해볼 때, 로지스틱 회귀는 Odds 관점에서 해석할 수 있다는 것이다.

> **Odds 란?** 어떤 시행에 대한 성공의 확률이 $$\pi$$라 하자. 이 때 성공에 대한 Odds는 $$\frac{\pi}{1-\pi}$$ 이다. 성공할 확률이 실패의 확률보다 몇배에 해당하는지를 보여주는 값이라 할 수 있으며 0 이상의 실수값을 갖는다. Odds가 1 이상의 값을 가지면 성공의 확률이 실패의 확률보다 크다고 할 수 있다. 만약 $$\pi=0.75$$라면, Odds는 3이다.

모델의 결과에 대한 해석이 가능하다는 점은 로지스틱 회귀가 갖는 장점으로 만약 $$\mathbf{X} = (x_{1},...,x_{k})$$ 이고 $$ d \leq k $$일 때, $$\text{exp}(\beta_{d})$$의 값은 나머지 변수가 고정인 상황에서 $$x_{d}$$가 1단위 변화할 때, Odds가 어떻게 변화하는지를 나타내는 값이라고 해석할 수 있다.








##### 상기 예제에 관련한 코드는 다음의 링크 1. [R코드](https://github.com/seolbluewings/rcode/blob/master/6.Probit_Regression.R) 2. [Python코드](https://github.com/seolbluewings/pythoncode/blob/master/6.Probit%20Regression.ipynb) 에서 확인할 수 있습니다.




