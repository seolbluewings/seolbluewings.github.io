---
layout: post
title:  "선형회귀 축소추정(Shrinkage Estimation)"
date: 2019-04-15
author: YoungHwan Seol
categories: Statistics
---

앞서 소개했던 선형 회귀의 LSE 값은 선형 불편(linear unbiased) 추정치들 중에서 분산이 가장 작다. 어떤 모수 $$\theta$$에 대한 추정치 $$\hat{\theta}$$의 평균제곱오차(MSE, mean squared error)는 다음과 같이 표현될 수 있다.

$$
\begin{align}
\text{MSE}(\hat{\theta}) &= \mathbb{E}(\hat{\theta}-\theta)^{2} \nonumber \\
&= \mathbb{E}(\hat{\theta}-\mathbb{E}(\hat{\theta})+\mathbb{E}(\hat{\theta})-\theta)^{2} \nonumber \\
&= (\hat{\theta}-\mathbb{E}(\hat{\theta}))^{2}+\mathbb{E}(\mathbb{E}(\hat{\theta})-\theta)^{2} \nonumber \\
&= \text{Var}(\hat{\theta})+\text{bias}^{2} \nonumber
\end{align}
$$

즉, MSE는 추정치의 분산과 편의(이하 bias)의 제곱의 합으로 이루어진다. LSE는 선형 불편추정량 중에서 분산이 가장 작다는 Gauss-Markov Theorm에 의해 선형 불편추정량들 중에서 가장 MSE가 작을 것이다.

그런데 불편성(unbiased)에서 벗어나 생각해보자. 추정값의 bias가 있으나 분산이 작아 전체 MSE 값이 LSE의 MSE값보다 작게 되는 추정치가 존재할 수 있을 것이다. 즉, bias가 존재하더라도 분산을 더 줄여서 전체 MSE 측면에서 더 작은 값을 갖게되는 추정값을 생각해볼 수 있을 것이다.

선형회귀를 활용한 예측을 진행하는 과정에서 MSE가 예측의 정확성과 관련이 있기 때문에 우리는 bias가 있을지라도 전체 MSE가 낮은 추정값을 찾아서 더 높은 예측력을 갖게 될 수 있을 것이다. 이러한 아이디어를 통해 만들어진 2가지 방법이 바로 능형회귀(이하 Ridge Regression)와 LASSO(Least Absolute Shrinkage and Selection Operator)이다.

#### Ridge Regression

먼저 소개했던 LSE 값은 다음과 같은 조건을 만족하는 $$\beta$$를 찾아내는 것이었다.

$$
\hat{\beta}_{LSE} = \text{argmin}_{\beta}(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)
$$

Ridge의 추정치는 $$ \bf{\beta}^{T}\bf{\beta} \leq t^{2} $$ 이란 제약조건 하에서 다음과 같이 구할 수 있다.

$$
\hat{\beta}_{Ridge} = \text{argmin}_{\beta}(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)
$$

이를 라그랑주 승수법에 의하면 다음과 같이 표현할 수 있고 이 식이 조금 더 익숙할 것이다.

$$
\hat{\beta}_{Ridge} = \text{argmin}_{\beta}\{(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)+\lambda \bf{\beta}^{T}\bf{\beta}\}
$$

앞서 

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
\text{where}\quad \bf{M} &= \bf{X}(\bf{X}^{T}\bf{X})^{-1}\bf{X}^{T} \nonumber
\end{align}
$$

우리의 목표는 앞서 수식에 표현한 것처럼 $$ \text{min}_{\beta} (\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta) $$ 을 구하는 것이며 다음과 같은 과정을 통해 구할 수 있다.

$$
\begin{align}
(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta) &= (\bf{Y}-\bf{MY}+\bf{MY}-\bf{X}\beta)^{T}(\bf{Y}-\bf{MY}+\bf{MY}-\bf{X}\beta) \nonumber \\

&= (\bf{Y}-\bf{MY})^{T}(\bf{Y}-\bf{MY}) + (\bf{MY}-\bf{X}\beta)^{T}(\bf{MY}-\bf{X}\beta) \nonumber
\end{align}
$$

식이 2가지 부분으로 나뉘는 것을 확인할 수 있고 두가지 값은 모두 0이상의 값을 가질 것이다. 여기서 첫번째 부분은 $$\beta$$가 존재하지 않는다. 따라서 이 값은  $$\beta$$에 의존하지 않는다.

따라서 $$(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)$$를 최소화시키는 것은 $$ (\bf{MY}-\bf{X}\beta)^{T}(\bf{MY}-\bf{X}\beta) $$를 최소화시키는 것과 동일하다. 왜냐면, 여기에는 $$\beta$$가 있기 때문이다. 따라서 여기서 $$\bf{MY} = \bf{X}\beta$$일 때, 최소값이 발생할 것이라는걸 알 수 있다. 이는 $$\beta$$의 LSE를 구하기 위한 필요충분조건에서 마주했던 값이다.

식을 전개하는 과정에서 교차항 부분에 대해서는 계산하지 않았는데 값이 0이 되기 때문이다. 그 이유는 다음과 같다.

$$
\begin{align}
(\bf{Y}-\bf{MY})^{T}(\bf{MY}-\bf{X}\beta) &= \bf{Y}^{T}(\bf{I}-\bf{M})\bf{MY}-\bf{Y}^{T}(\bf{I}-\bf{M})\bf{X}\beta = 0\nonumber
\\
(\bf{I}-\bf{M})\bf{M} &= 0 \nonumber \\
(\bf{I}-\bf{M})\bf{X} &= 0 \nonumber
\end{align}
$$

$$\bf{M}$$값을 위에서 $$\bf{X}$$로 표현했던 것을 대입해보면 첫번째 부분이 0이되는 것을 확인할 수 있고 두번째 부분이 0이 되는 것은 기하학적인 부분에서 설명했던 것을 활용하면 이해할 수 있다. M이 X가 Span하는 공간으로 Orthogonal Projection 시키는 행렬인데 X를 X가 Span하는 공간으로 Orthogonal Projection 시켜도 X 그 자체이기 때문이다.

#### 오차항 가정으로 인한 결과들

잘 생각해보면 지금까지 우리는 $$\beta$$의 LSE를 구하는 과정에서 오차항이 가우시안 분포를 따른다는 성질을 전혀 사용하지 않았다. 지금부터는 오차항이 가우시안 분포 $$\mathcal{N}(\bf{0},\sigma^{2}\bf{I})$$를 따른다는 가정을 통해 얻을 수 있는 성질들에 대해 이야기할 것이다.

#### 1.오차에 대한 불편추정량

일단 가우시안 분포 가정이 이루어졌을 때, $$\sigma^{2}$$에 대한 불편 추정량(unbiased estimator)을 어떻게 구할 수 있는지 생각해보자. 이 때, $$r(\bf{X})=r$$이라 가정하자.

$$\bf{Y}$$의 추정값인 $$\hat{\bf{Y}}$$는 $$\hat{\bf{Y}} = \bf{MY} = \bf{X}\hat{\beta}$$ 과 같이 구할 수 있다. 그렇다면, 잔차 $$\hat{\epsilon}$$ 는 $$\bf{Y}-\bf{MY}$$로 표현할 수 있다. 따라서 $$\bf{Y} = \bf{MY} + (\bf{I}-\bf{M})\bf{Y}$$ 로 표현될 수 있다.

기존 $$\bf{Y} = \bf{X}\beta + \epsilon $$ 식의 양변에 $$(\bf{I}-\bf{M})$$을 곱해보자. $$(\bf{I}-\bf{M})\bf{X}=0$$ 이기 때문에 $$(\bf{I}-\bf{M})\bf{Y}$$의 값은 $$\epsilon$$에 의존한다. $$\sigma^{2}$$ 값은 오차항의 성질을 결정짓기 때문에 우리는 $$\sigma^{2}$$을 추정하는 과정에서 $$(\bf{I}-\bf{M})\bf{Y}$$를 이용해야 한다. Quadratic Form 특성에 의해, 다음과 같은 식이 성립한다.

$$
\mathbb{E}[\bf{Y}^{T}(\bf{I}-\bf{M})\bf{Y}] = \text{tr}[\sigma^{2}(\bf{I}-\bf{M})] + \beta^{T}\bf{X}^{T}(\bf{I}-\bf{M})\bf{X}\bf{\beta}
$$

여기에서 $$(\bf{I}-\bf{M})\bf{X}=0$$ 이라는 것과 $$ \text{tr}[\sigma^{2}(\bf{I}-\bf{M})] = \sigma^{2}\text{tr}(\bf{I}-\bf{M})= \sigma^{2}(n-r)$$
이라는 것을 활용하면 다음과 같이 정리될 수 있다.

$$
\begin{align}
\mathbb{E}[\bf{Y}^{T}(\bf{I}-\bf{M})\bf{Y}] &= \sigma^{2}(n-r) \nonumber \\
\mathbb{E}[\bf{Y}^{T}(\bf{I}-\bf{M})\bf{Y}/(n-r)] &= \sigma{^2}
\end{align}
$$

$$\sigma^{2}$$의 불편추정량은 $$ \bf{Y}^{T}(\bf{I}-\bf{M})\bf{Y}/(n-r) $$ 으로 구할 수 있고 이를 MSE(mean squared error)라고 부른다. $$n-r$$로 나누기 이전의 $$ \bf{Y}^{T}(\bf{I}-\bf{M})\bf{Y} $$ 값은 SSE (sum of squares for error)라고 불린다.


#### 2.$$\beta$$에 대한 최대 가능도 값이 곧 최소 제곱법으로 추정하는 값이다.

입력 데이터 집합 $$\bf{X}_{N} = \{x_{1},...,x_{N}\}$$ 타깃변수 집합 $$\bf{y}_{N} =\{y_{1},...,y_{N}\}$$ 과 같이 존재한다고 생각해보자. 이 타깃변수의 데이터 포인트 $$(y_{1},...,y_{n})$$ 가 $$ \text{P}(\bf{y}\mid\bf{X},\beta,\sigma^{2}) $$ 로부터 독립적으로 추출된다는 가정을 하자. 그렇다면 다음과 같이 가능도(Likelihood)를 계산할 수 있다.

$$
\text{P}(\bf{y}\mid\bf{X},\beta,\sigma^{2}) = \prod_{i=1}^{N}\mathcal{N}(y_{i}\mid x_{i}^{T}\beta,\sigma^{2})
$$

앞으로의 과정에서 입력변수 $$\bf{X}$$의 분포에 대해서는 관심을 두지 않으므로 모든 과정에서 $$\bf{X}$$ 를 생략하더라도 조건부 변수로 설정되어 있다고 생각하자. 그리고 앞서 언급한 가능도에 로그를 취하면 다음과 같이 표현할 수 있다.

$$
\begin{align}
\log{\text{P}(\bf{y}\mid\beta,\sigma^{2})} &= \sum_{i=1}^{N}\log{\mathcal{N}(y_{i}\mid x_{i}^{T}\beta,\sigma^{2})} \nonumber \\
&= -\frac{N}{2}\log{2\pi}-\frac{N}{2}\log{\sigma^{2}}-\frac{1}{2\sigma^{2}}(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)
\end{align}
$$

로그 가능도를 구하였으니 MLE 방식을 통하여 우리는 $$\beta$$와 $$\sigma^{2}$$ 에 대한 추정을 시도할 수 있다. 그런데 주목할 사실은 지금처럼 노이즈를 가우시안 분포로 가정한 경우 $$\hat{\beta_{\text{MLE}}}$$와 $$\hat{\beta_{\text{LSE}}}$$ 결과가 동일하다는 것이다.

위의 로그 가능도에서 모수 $$\beta$$에 대한 관련있는 부분만 고려해보자. 그렇다면 로그 가능도의 앞 2가지 부분을 제외할 수 있고 $$ -\frac{1}{2\sigma^{2}}(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta) $$ 부분만 남는 것을 확인할 수 있다. 이를 최대화시키는 값을 찾는 것이 모수 $$\beta$$의 MLE값을 구하는 것이다. 그런데 잘 생각해보면, 이 값을 최대화하는 것은 $$ (\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta $$ 값을 최소화시키는 것과 동일하다. 이는 최소제곱법 방법과 동일하다.

한편, $$\sigma^{2}$$의 경우는 MLE로 추정하는 방식이 LSE로 추정하는 방식과 다르다. 앞서 LSE 방법을 활용할 때는 $$\hat{\sigma^{2}} = \bf{Y}^{T}(\bf{I}-\bf{M})\bf{Y}/(n-r)$$ 이었으나 MLE 방식에서는 $$\hat{\sigma^{2}} = \bf{Y}^{T}(\bf{I}-\bf{M})\bf{Y}/n$$ 으로 값이 추정된다.










