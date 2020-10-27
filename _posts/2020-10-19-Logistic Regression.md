---
layout: post
title:  "로지스틱 회귀(Logistic Regression)"
date: 2020-10-19
author: seolbluewings
categories: 선형모델
---

[작성중...]

로지스틱 회귀는 프로빗 모델처럼 반응변수 범주형인 케이스에서 활용할 수 있는 방법이다. 로지스틱 회귀는 새로운 변수(X)가 주어졌을 때, 반응 변수가 각 범주에 속할 확률이 얼마인지를 추정하며, 추정된 확률에 따라 반응 변수의 Class를 분류하게 된다.

일단 반응 변수의 클래스가 2가지인 경우, 즉 $$ y \in \{0,1\}$$ 로 표현되는 경우를 생각해보자.

<center>![LR](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Logistic.png?raw=true){:width="70%" height="70%"}</center>

그림처럼 성공/실패가 명확하게 구분되는 케이스에서 기존의 선형 모델을 그대로 적용하는 것은 무리가 있다. 비선형 함수를 활용하여 성공/실패, $$y=0$$ 또는 $$y=1$$을 구분하는 것이 바람직하다고 할 수 있다. 그림의 우측에서 활용되는 비선형 함수는 시그모이드 함수이며, 이처럼 시그모이드 함수를 연결함수로 활용하여 데이터를 적합시키는 것을 로지스틱 회귀라고 부른다.

$$
\begin{align}
\pi(\mathbf{X}) &= p(y=1\mid\mathbf{X}) \nonumber \\
\log{\frac{\pi(\mathbf{X})}{1-\pi(\mathbf{X})}} &= \mathbf{X}\beta \nonumber \\
\pi(\mathbf{X}) &= \frac{e^{\mathbf{X}\beta}}{1+e^{\mathbf{X}\beta}} \nonumber
\end{align}
$$

여기서 로지스틱 회귀의 중요한 특징을 하나 발견할 수 있다. 수식의 형태를 고려해볼 때, 로지스틱 회귀는 Odds 관점에서 해석할 수 있다는 것이다.

> **Odds 란?** 어떤 시행에 대한 성공의 확률이 $$\pi$$라 하자. 이 때 성공에 대한 Odds는 $$\frac{\pi}{1-\pi}$$ 이다. 성공할 확률이 실패의 확률보다 몇배에 해당하는지를 보여주는 값이라 할 수 있으며 0 이상의 실수값을 갖는다. Odds가 1 이상의 값을 가지면 성공의 확률이 실패의 확률보다 크다고 할 수 있다. 만약 $$\pi=0.75$$라면, Odds는 3이다.

모델의 결과에 대한 해석이 가능하다는 점은 로지스틱 회귀가 갖는 장점으로 만약 $$\mathbf{X} = (x_{1},...,x_{k})$$ 이고 $$ d \leq k $$일 때, $$\text{exp}(\beta_{d})$$의 값은 나머지 변수가 고정인 상황에서 $$x_{d}$$가 1단위 변화할 때, Odds가 어떻게 변화하는지를 나타내는 값이라고 해석할 수 있다.

따라서 로지스틱 회귀는 선형 모델의 예측 결과를 활용하여 데이터의 로그 Odds에 근사하는 것이다. 여기서 주의할 부분은 이름은 회귀지만, 실제로 행하는 작업은 분류라는 것이다. 더불어 로지스틱 회귀는 단순하게 결과를 0,1로만 return해준다기 보다는 각 Class로 할당될 확률(0,1로 결정될 확률)을 근사적으로 보여준다고 할 수 있다.

앞서 나열했던 수식을 활용하면 다음과 같은 관계가 나타남을 확인할 수 있다.

$$
\begin{align}
\pi(\mathbf{X}) &= p(y=1\mid\mathbf{X}) \nonumber \\
p(y=1\mid\mathbf{X}) &= \pi(\mathbf{X}) = \frac{e^{\mathbf{X}\beta}}{1+e^{\mathbf{X}\beta}} \nonumber \\
p(y=0\mid\mathbf{X}) &= 1-\pi(\mathbf{X}) = \frac{1}{1+e^{\mathbf{X}\beta}} \nonumber
\end{align}
$$

#### 로지스틱 회귀 추정

로지스틱 회귀문제에서 parameter $$\beta$$에 대한 추정은 로지스틱 회귀의 중요한 이슈이다. 기존의 선형 모델에서는 오차제곱합을 최소화시키는 parameter $$\beta$$를 구했다. 이는 Convex 함수이기 때문에 Global한 minimal을 비교적 쉽게 찾을 수 있다.

$$\text{argmin}_{\beta} (Y-X\beta)^{T}(Y-X\beta)$$

비슷한 방식을 취한다고 한다면, 로지스틱 회귀의 경우는 아래와 같을 것이다. h는 연결함수인 것으로 받아들이자.

$$\text{argmin}_{\beta} (Y-h(X\beta))^{T}(Y-h(X\beta))$$

그런데 이 수식은 앞선 단순 선형회귀 모델의 경우와 달리 Convex하지 않게 된다. non-Convex한 함수가 되어 Global minimal을 찾기가 어려워진다. 따라서 로지스틱 회귀에서 parameter $$\beta$$ 를 추정하는 것이 중요한 이슈이다.

기존 선형회귀 모델에서 최소제곱추정량(Least Square Estimator, LSE)말고도 최대우도추정량(Maximum Likelihood Estimator, MLE)가 존재했던 것을 생각해보자. 로지스틱 회귀에서는 MLE을 사용하여 회귀계수 $$\beta$$를 추정한다.




##### 상기 예제에 관련한 코드는 다음의 링크 1. [R코드](https://github.com/seolbluewings/rcode/blob/master/6.Probit_Regression.R) 2. [Python코드](https://github.com/seolbluewings/pythoncode/blob/master/6.Probit%20Regression.ipynb) 에서 확인할 수 있습니다.




