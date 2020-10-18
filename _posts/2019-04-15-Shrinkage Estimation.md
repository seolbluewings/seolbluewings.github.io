---
layout: post
title:  "선형회귀 축소추정(Shrinkage Estimation)"
date: 2019-04-15
author: YoungHwan Seol
categories: 선형 모델(Linear Model)
---

앞서 소개했던 선형 회귀의 LSE 값은 선형 불편(linear unbiased) 추정치들 중에서 분산이 가장 작다. 어떤 모수 $$\theta$$에 대한 추정치 $$\hat{\theta}$$의 평균제곱오차(MSE, mean squared error)는 다음과 같이 표현될 수 있다.

$$
\begin{align}
\text{MSE}(\hat{\theta}) &= \mathbb{E}(\hat{\theta}-\theta)^{2} \nonumber \\
&= \mathbb{E}(\hat{\theta}-\mathbb{E}(\hat{\theta})+\mathbb{E}(\hat{\theta})-\theta)^{2} \nonumber \\
&= \mathbb{E}(\hat{\theta}-\mathbb{E}(\hat{\theta}))^{2} +(\mathbb{E}(\hat{\theta})-\theta)^{2} \nonumber \\
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

Ridge의 추정치는 $$ \mid\beta\mid_{2}^{2} \leq t^{2} $$, 즉 L2 Norm이 특정값보다 작다는 제약조건 하에서 다음과 같이 구할 수 있다.

$$
\hat{\beta}_{Ridge} = \text{argmin}_{\beta}(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)
$$

이를 라그랑주 승수법에 의하면 다음과 같이 표현할 수 있다. 여기서 $$\lambda \geq 0$$ 이다.

$$
\hat{\beta}_{Ridge} = \text{argmin}_{\beta}\{(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)+\lambda |\beta|_{2}^{2} \}
$$

이는 LSE를 구하는 과정에 $$ \lambda \mid\beta\mid_{2}^{2} $$ 라는 Penalty Term이 들어가는 것이며 이 Penalty Term까지 들어간 부분을 포함해 전체를 최소화시키는 것을 의미한다. 이 Penalty Term을 정규화항이라 부르기도 하며  데이터에 의해 지지되지 않는 한 parameter 값이 0을 향해 감소하기 때문에 이를 매개변수 축소(parameter shrinkage)라 부른다.

기존 LSE의 추정값은 $$ \hat{\beta}_{LSE} = (\bf{X}^{T}\bf{X})^{-1}\bf{X}^{T}\bf{Y} $$ 인데 Ridge Regression의 경우 $$ \hat{\beta}_{Ridge} =  (\bf{X}^{T}\bf{X}+\lambda\bf{I})^{-1}\bf{X}^{T}\bf{Y} $$ 의 형태로 구할 수 있다.

#### LASSO

Ridge Regression은 축소된 추정값을 주지만, 이를 통해 변수 선택까지 진행할 수는 없다. 따라서 고차원 자료의 경우 최종 모형에 대한 해석이 쉽지 않을 수 있다. 그래서 지금부터는 축소된 추정값 뿐만 아니라 변수선택을 통해 예측력을 향상시키는 LASSO에 대해 이야기하고자 한다.

LASSO는 제약조건 $$ \mid\beta\mid_{1} \leq t $$ 즉 L1 Norm 이 특정값 이하의 값을 갖는다는 조건 하에 다음의 식을 만족하는 $$\beta$$를 구하는 것이다.

$$
\hat{\beta}_{LASSO} = \text{argmin}_{\beta}(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)
$$

이를 라그랑주 승수법에 의하면 다음과 같이 표현할 수 있다. 여기서 $$\lambda \geq 0$$ 이다.

$$
\hat{\beta}_{LASSO} = \text{argmin}_{\beta}\{(\bf{Y}-\bf{X}\beta)^{T}(\bf{Y}-\bf{X}\beta)+\lambda |\beta|_{1}\}
$$

이 때, $$\lambda$$값을 충분히 크게 설정하면 몇몇 계수 $$\beta_{j}$$ 값은 0이 될 것이다. 이 때, $$\beta_{j}$$가 0이 된 항에 매칭되는 $$\bf{X}$$의 변수는 더 이상 사용되지 않는다.

![Shrinkage](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/LASSO.png?raw=true)

Ridge 와 LASSO의 차이는 Penalty Term이 L2 Norm에서 L1 Norm으로 바뀐다는 것이다. 위의 그림은 Ridge와 LASSO의 차이를 보여주는 그림으로 보여주는 대표적인 예시이다. 이 그림에서 등고선은 $$\hat{\beta}_{LSE}$$ 을 중심으로 하는 오차제곱합의 등고선이다.

음영처리된 부분은 제약조건인 $$ \mid\beta_{1}\mid + \mid\beta_{2}\mid \leq t $$ 와 $$ \beta_{1}^{2} + \beta_{2}^{2} \leq t^{2} $$ 을 만족시키는 영역을 표현한다. 각 $$\beta_{j}$$의 추정치는 등고선과 음영처리된 영역이 만나는 점으로 주어지게 된다.

LASSO의 경우 제약조건을 나타내는 영역이 정사각형 형태이므로 추정값이 모서리제 닿아 계수가 0이될 가능성이 높다. LASSO는 설명력이 없는 입력변수들의 계수를 0으로 추정함으로써 자동적으로 변수선택이 이루어진다.

