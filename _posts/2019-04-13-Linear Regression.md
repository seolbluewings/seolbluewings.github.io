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

노이즈가 가우시안 분포라고 가정하는 것은 $$\bf{X}$$가 주어진 상황에서 $$\bf{y}$$의 조건부 분포가 단봉 분포라는 것을 내포하는데 때로는 이러한 가정이 적절하지 않을 수도 있다는 것을 유념해야 한다. 어쨌든 앞으로 논의를 이어나가는 과정에서는 다음과 같이 각 변수끼리 독립적인 가우시안 노이즈를 가정할 것이다.

$$
\bf{Y} = \bf{X}\beta + \epsilon \quad \mathbb{E}(\epsilon) = 0 \quad \text{Cov}(\epsilon) = \sigma^{2}\bf{I}
$$

최고제곱법을 활용하는 우리의 시도는 $$\mathbb{E}(\bf{Y})$$ 값을 구하는 것이며 이는 기하학적으로 생각해보면, $$\bf{X}$$가 Span하는 Subspace에서 $$\bf{Y}$$ 라는 벡터와 가장 근접한 벡터를 찾는 과정이다. $$\bf{X}\hat{\beta}$$가 $$\bf{X}$$가 Span하는 Subspace 내에 존재하는 벡터라면, 이 때 $$\hat{\beta}$$는 $$\beta$$의 LSE(Least Square Estimate)라 불린다. 이를 수식으로 표현하자면 다음과 같다.

$$
\text{min}_{\beta} (\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta) = (\bf{Y}-\bf{X}\hat{\beta})^{T}(\bf{Y}-\bf{X}\hat{\beta})
$$

벡터 $$\bf{Y}$$를 X가 Span하는 Subspace로 Orthogonal Projection 시키는 행렬을 $$\bf{M}$$이라 한다면, 다음과 같은 수식이 성립하며 이 식은 $$\hat{\beta}$$가 $$\beta$$의 LSE가 되는 필요충분조건이다.

$$
\begin{align}
\bf{MY} &= \bf{X}\hat{\beta} \nonumber \\
\text{where}\quad \bf{M} &= (\bf{X}^{T}\bf{X})^{-1}\bf{X}^{T} \nonumber
\end{align}
$$

우리의 목표는 앞서 수식에 표현한 것처럼 $$ \text{min}_{\beta} (\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta) $$ 을 구하는 것이며 다음과 같은 과정을 통해 구할 수 있다.

$$
\begin{align}
(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta) &= (\bf{Y}-\bf{MY}+\bf{MY}-\bf{X}\beta)^{T}(\bf{Y}-\bf{MY}+\bf{MY}-\bf{X}\beta) \nonumber \\

&= (\bf{Y}-\bf{MY})^{T}(\bf{Y}-\bf{MY}) + (\bf{MY}-\bf{X}\beta)^{T}(\bf{MY}-\bf{X}\beta) \nonumber
\end{align}
$$

식이 2가지 부분으로 나뉘는 것을 확인할 수 있고 두가지 값은 모두 0이상의 값을 가질 것이다. 여기서 첫번째 부분은 $$\beta$$가 존재하지 않는다. 따라서 이 값은  $$\beta$$에 의존하지 않는다. 따라서 $$(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)$$를 최소화시키는 것은 $$ (\bf{MY}-\bf{X}\beta)^{T}(\bf{MY}-\bf{X}\beta) $$를 최소화시키는 것과 동일하다. 왜냐면, 여기에는 $$\beta$$가 있기 때문이다. 따라서 여기서 $$\bf{MY} = \bf{X}\beta$$일 때, 최소값이 발생할 것이라는걸 알 수 있다. 이는 $$\beta$$의 LSE를 구하기 위한 필요충분조건에서 마주했던 값이다.

식을 전개하는 과정에서 교차항 부분에 대해서는 계산하지 않았는데 값이 0이 되기 때문이다. 그 이유는 다음과 같다.

$$
\begin{align}
(\bf{Y}-\bf{MY})^{T}(\bf{MY}-\bf{X}\beta) &= \bf{Y}^{T}(\bf{I}-\bf{M})\bf{MY}-\bf{Y}^{T}(\bf{I}-\bf{M})\bf{X}\beta = 0\noumber
\\
\text{Since} \quad (\bf{I}-\bf{M})\bf{M} &= 0 \nonumber \\
(\bf{I}-\bf{M})\bf{X} &= 0 \nonumber
\end{align}
$$

$$\bf{M}$$값을 위에서 $$\bf{X}$$로 표현했던 것을 대입해보면 첫번째 부분이 0이되는 것을 확인할 수 있고 두번째 부분이 0이 되는 것은 기하학적인 부분에서 설명했던 것을 활용하면 이해할 수 있다. M이 X가 Span하는 공간으로 Orthogonal Projection 시키는 행렬인데 X를 X가 Span하는 공간으로 Orthogonal Projection 시켜도 X 그 자체이기 때문이다.

#### 오차항 가정이 만들어내는 차이

잘 생각해보면 지금까지 우리는 $$\beta$$의 LSE를 구하는 과정에서 오차항이 가우시안 분포를 따른다는 성질을 전혀 사용하지 않았다. 지금부터는 오차항이 가우시안 분포 $$\mathcal{N}(\bf{0},\sigma^{2}\bf{I})$$를 따른다는 가정을 통해 얻을 수 있는 성질들에 대해 이야기할 것이다.

일단 가우시안 분포 가정이 이루어졌을 때, $$\sigma^{2}$$에 대한 불편 추정량(unbiased estimator)을 어떻게 구할 수 있는지 생각해보자.


#### 최대 가능도

입력 데이터 집합 $$\bf{X}_{N} = \{x_{1},...,x_{N}\}$$ 타깃변수 집합 $$\bf{y}_{N} =\{y_{1},...,y_{N}\}$$ 과 같이 존재한다고 생각해보자. 이 타깃변수의 데이터 포인트 $$(y_{1},...,y_{n})$$ 가 $$ \text{P}(\bf{y}\mid\bf{X},\beta,\sigma^{2}) $$ 로부터 독립적으로 추출된다는 가정을 하자. 그렇다면 다음과 같이 가능도(Likelihood)를 계산할 수 있다.

$$
\text{P}(\bf{y}\mid\bf{X},\beta,\sigma^{2}) = \prod_{i=1}^{N}\mathcal{N}(y_{i}\mid x_{i}^{T}\beta,\sigma^{2})
$$

앞으로의 과정에서 입력변수 $$\bf{X}$$의 분포에 대해서는 관심을 두지 않으므로 모든 과정에서 $$\bf{X}$$ 를 생략하더라도 조건부 변수로 설정되어 있다고 생각하자. 그리고 앞서 언급한 가능도에 로그를 취하면 다음과 같이 표현할 수 있다.

$$
\begin{align}
\log{\text{P}(\bf{y}\mid\beta,\sigma^{2})} &= \sum_{i=1}^{N}\log{\mathcal{N}(y_{i}\mid x_{i}^{T}\beta,\sigma^{2})} \nonumber \\
&= -\frac{N}{2}\log{2\pi}-\frac{N}{2}\log{\sigma^{2}}-\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_{i}-x_{i}^{T}\beta)^{2}
\end{align}
$$

로그 가능도를 구하였으니 MLE 방식을 통하여 우리는 $$\beta$$와 $$\sigma^{2}$$ 에 대한 추정을 시도할 수 있다. 그런데 주목할 사실은 지금처럼 노이즈를 가우시안 분포로 가정한 경우 $$\hat{\beta_{\text{MLE}}}$$와 $$\hat{\beta_{\text{LSE}}}$$ 결과가 동일하다는 것이다.

위의 로그 가능도에서 모수 $$\beta$$에 대한 관련있는 부분만 고려해보자. 그렇다면 로그 가능도의 앞 2가지 부분을 제외할 수 있고 $$ -\frac{1}{2\sigma^{2}}\sum_{i=1}^{N}(y_{i}-x_{i}^{T}\beta)^{2} $$ 부분만 남는 것을 확인할 수 있다. 이를 최대화시키는 값을 찾는 것이 모수 $$\beta$$의 MLE값을 구하는 것이다. 그런데 잘 생각해보면, 이 값을 최대화하는 것은 $$ \sum_{i=1}^{N}(y_{i}-x_{i}^{T}\beta)^{2} $$ 값을 최소화시키는 것과 동일하다. 이는 최소제곱법 방법과 동일하다.












