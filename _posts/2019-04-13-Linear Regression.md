---
layout: post
title:  "선형 회귀 모델(Linear Regression)"
date: 2019-04-13
author: YoungHwan Seol
categories: Bayesian
---

회귀분석이란 P차원 벡터($$\bf{X}$$)가 입력(input)될 때, 그에 대응하는 연속형 타깃 변수(target) $$\bf{y}$$를 예측하는 것이다. N개의 관측값 $$X_{N}$$이 있다고 하자. 이에 대응하는 변수 $$y_{N}$$이 훈련 집합으로 존재한다고 하자. 이 때, 선형회귀모델의 목표는 새로운 변수 $$X_{\text{new}}$$의 종속변수인 $$y_{\text{new}}$$를 예측하는 것이며 최종적으로 $$\text{P}(y_{\text{new}}\mid X_{\text{new}})$$ 의 분포를 모델링하는 것이다. 우리는 이 예측분포를 통해서 $$X_{\text{new}}$$에 대한 $$y_{\text{new}}$$의 불확실성을 표현할 수 있다.

#### '선형' 회귀(Linear Regression)

가장 단순한 형태의 선형 회귀 모델은 입력 변수들의 선형 결합을 바탕으로 한 모델로 다음과 같이 표현할 수 있다.

$$\bf{Y} = \bf{X}\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0,\sigma^{2})$$

여기서 짚고 넘어가야할 부분은 무엇에 관한 '선형'이냐는 것이다. 로지스틱 회귀의 경우는 위와 달리 $$\text{logit}(\bf{y}) = \bf{X}\beta $$ 로 표기되는데 이 역시도 선형 모델이라고 불린다. 그래서 입력 변수 $$\bf{X}$$에 대한 선형 함수(결합)가 아닌 $$\beta$$에 대한 선형 함수라고 할 수 있다. 그래서 선형 모델인 것이다. 물론 $$\bf{X}\beta$$의 경우는 입력변수 $$\bf{X}$$의 선형 함수이기도 하다.

#### 최소 제곱법

앞서 표기한 것처럼 $$\bf{Y}$$는 $$\bf{X}\beta$$와 가우시안 노이즈 $$\epsilon$$의 합으로 이루어진다고 하자.

그렇다면, $$\text{P}(\bf{y}\mid\bf{X},\beta,\sigma^{2})$$의 분포는 $$\mathcal{N}(\bf{y}\mid\bf{X}\beta, \sigma^{2})$$ 로 표현될 수 있다.

손실함수를 다음과 같이 오차제곱합 함수 $$\text{L}(y,\text{f}(x)) = \{y-\text{f}(x)\}^2 $$ 의 형태를 가정한다면, 평균적인 손실은 다음과 같이 표현할 수 있을 것이다.

$$
\begin{align}
\mathbb{E}[L] &= \int\int L(y,\text{f}(x))\text{P}(\bf{X},\bf{y})d\bf{X}d\bf{y} \nonumber \\
&= \int\int \{y-\text{f}(x)\}^{2}\text{P}(\bf{X},\bf{y})d\bf{X}d\bf{y} \nonumber
\end{align}
$$

최소제곱법의 목표는 결국 $$\mathbb{E}[L]$$ 를 최소화하는 $$\text{f}(\bf{X})$$를 선택하는 것이다. 여기서 변분법을 적용하면, 다음과 같이 표기할 수 있다.

$$ 2\int L(y,\text{f}(x))\text{P}(\bf{X},\bf{y})d\bf{y} = 0 $$

이를 정리하면 다음과 같을 것이며, 즉 손실함수를 최소화하는 함수 $$\text{f}(\bf{X})$$는 $$\bf{X}$$가 주어졌을 때, $$\bf{y}$$의 조건부 평균이다.

$$
\begin{align}
\text{f}(\bf{X}) &= \frac{ \int\bf{y}\text{P}(\bf{X},\bf{y})d\bf{y}}{ \text{P}(\bf{X}) } \nonumber \\
&= \int \bf{y}\text{P}(\bf{y}\mid\bf{X})d\bf{y} = \mathbb{E}_{\text{y}}[\bf{y}\mid\bf{X}] \nonumber \\
\mathbb{E}_{\text{y}}[\bf{y}\mid\bf{X}] &= \int \bf{y}\text{P}(\bf{y}\mid\bf{X})d\bf{y} = \bf{X}\beta \nonumber
\end{align}
$$

다음과 같이 노이즈가 가우시안 분포라고 가정하는 것은 $$\bf{X}$$가 주어진 상황에서 $$\bf{y}$$의 조건부 분포가 단봉 분포라는 것을 내포하는데 때로는 이러한 가정이 적절하지 않을 수도 있다는 것을 유념해야 한다.

#### 최대 가능도

입력 데이터 집합 $$\bf{X}_{N} = \{x_{1},...,x_{N}\}$$ 타깃변수 집합 $$\bf{y}_{N} =\{y_{1},...,y_{N}\}$$ 과 같이 존재한다고 생각해보자. 이 타깃변수의 데이터 포인트 $$(y_{1},...,y_{n})$$ 가 $$ \text{P}(\bf{y}\mid\bf{X},\beta,\sigma^{2}) $$ 로부터 독립적으로 추출된다는 가정을 하자. 그렇다면 다음과 같이 가능도(Likelihood)를 계산할 수 있다.

$$
\text{P}(\bf{y}\mid\bf{X},\beta,\sigma^{2}) = \prod_{i=1}^{N}\mathcal{N}(y_{i}\mid x_{i}^{T}\beta,\sigma^{2})
$$

앞으로의 과정에서 입력변수 $$\bf{X}$$의 분포에 대해서는 관심을 두지 않으므로 모든 과정에서 $$\bf{X}$$ 를 생략하더라도 조건부 변수로 설정되어 있다고 생각하자. 그리고 앞서 언급한 가능도에 로그를 취하면 다음과 같이 표현할 수 있다.

$$
\begin{align}
\log{\text{P}(\bf{y}\mid\beta,\sigma^{2})} &= \sum_{i=1}^{N}\log{\mathcal{N}(y_{i})\mid x_{i}^{T}\beta,\sigma^{2})} \nonumber \\
&= -\frac{N}{2}\log{2\pi}-\frac{N}{2}\log{\sigma^{2}}-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_{i}-x_{i}^{T}\beta)^{2}
\end{align}
$$

로그 가능도를 구하였으니 MLE 방식을 통하여 우리는 $$\beta$$와 $$\sigma^{2}$$ 에 대한 추정을 시도할 수 있다. 그런데 주목할 사실은 지금처럼 노이즈를 가우시안 분포로 가정한 경우 $$\beta_{\text{MLE}}$$와 $$\beta_{\text{LSE}}$$ 결과가 동일하다는 것이다.

위의 로그 가능도에서 모수 $$\beta$$에 대한 관련있는 부분만 고려해보자. 그렇다면 로그 가능도의 앞 2가지 부분을 제외할 수 있고 $$ -\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_{i}-x_{i}^{T}\beta)^{2} $$ 부분만 남는 것을 확인할 수 있다. 이를 최대화시키는 값을 찾는 것이 모수 $$\beta$$의 MLE값을 구하는 것이다. 그런데 잘 생각해보면, 이 값을 최대화하는 것은 $$ \sum_{i=1}^{N}(y_{i}-x_{i}^{T}\beta)^{2} $$ 값을 최소화시키는 것과 동일하다. 이는 최소제곱법 방법과 동일하다.













