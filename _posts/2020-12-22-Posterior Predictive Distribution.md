---
layout: post
title:  "Posterior Predictive Distribution"
date: 2020-12-22
author: seolbluewings
comments : true
categories: Bayesian
---

앞서 소개했던 Bayesian Linear Regression [포스팅](https://seolbluewings.github.io/bayesian/2019/04/22/Bayesian-Linear-Regression.html)을 통해서 우리는 기존의 Frequentist들의 회귀계수 $$\beta$$에 대한 추정방법과 다른 Bayesian 방식을 학습하였다. 그러나 분석 목적에 따라 $$\beta$$를 추정하는 것보다 $$\beta$$ 값을 이용하여 새로운 독립적인 데이터 $$x_{new}$$가 주어졌을 때, 예측값인 $$y_{new}$$ 를 구하는 것이 더 중요할 수도 있다. 앞으로 편의상 $$y$$에 대한 예측값은 $$\tilde{y}$$로 표현하겠다.

$$\tilde{y}$$를 추정하기 위해서 Bayesian이 취할 수 있는 방식은 $$\tilde{y}$$에 대한 분포를 구하는 것일거다. 이를 $$\tilde{y}$$에 대한 예측 분포(Predictive Distribution)이라 부르며 일반적으로 Posterior Predictive Distribution이라 말한다.

#### Posterior Predictive Distribution

$$\tilde{y}$$의 Posterior Predictive Distribution에 대한 일반적인 수식은 다음과 같다.

$$
\begin{align}
\it{p}(\tilde{y}\vert y) &= \int \it{p}(\tilde{y},\theta \vert y)d\theta \nonumber \\
&= \int \it{p}(\tilde{y}\vert\theta,y)p(\theta\vert y)d\theta \nonumber \\
&= \int \it{p}(\tilde{y}\vert\theta)p(\theta\vert y)d\theta \nonumber
\end{align}
$$

2번째 줄에서 3번째 줄로 넘어가는 과정에서 한가지 가정이 성립되어야 하는데 그 가정은 바로 $$\theta$$가 주어진 상황에서 $$\tilde{y}$$와 $$y$$가 conditional independence 하다는 것이다. conditional iondependence에 대한 개념은 이 [포스팅](https://seolbluewings.github.io/bayesian/2019/07/25/Bayesian-Network(1).html)에서 확인할 수 있다.

수리적으로 적분을 통해 Posterior Predictive Distribution을 구하는 것은 결코 쉬운 일이 아니다. 적분을 통해 우리가 아는 형태의 Posterior Predicitve Distribution이 나오는 경우는 드물다고 봐야한다. 다행스럽게도 정규 분포의 경우, 이 적분을 수행하지 않고도 예측 분포를 구할 수 있다.

#### 정규 분포에서의 Posterior Predictive Distribution

이전에 우리가 Bayesian Linear Regression을 수행했던 결과를 떠올려 보자. $$\beta$$에 대한 사후분포의 형태는 다음과 같았다.

$$ \it{p}(\beta \vert \sigma, y) \sim \mathcal{N}(\mu_{\beta},\Sigma_{\beta})$$

여기서 $$\Sigma_{\beta} = \left(\frac{1}{\sigma^{2}}X^{T}X +\Sigma_{0}^{-1} \right)^{-1}$$ 이며 $$ \mu_{\beta} = \Sigma_{\beta}\left(\frac{1}{\sigma^{2}}X^{T}y+\Sigma^{-1}_{0}\beta_{0}\right)$$ 이다.

새로운 $$\tilde{y}$$에 대한 분포는 다음과 같을 것이다.

$$\tilde{y}\vert \mu_{\beta},\Sigma_{\beta} \sim \mathcal{N}(\tilde{x}^{T}\beta,\sigma^{2})$$

그렇다면, $$\tilde{y}-\tilde{x}^{T}\beta$$는 $$\mathcal{N}(0,\sigma^{2})$$의 분포를 따를 것이고 이는 $$\beta$$에 의존하지 않는다. 따라서 $$\tilde{y}-\tilde{x}^{T}\beta$$와 $$\beta$$는 독립적이다.

$$\tilde{y}-\tilde{x}^{T}\beta \sim \mathcal{N}(0,\sigma^{2})$$ 와 $$\tilde{x}^{T}\beta\vert y \sim \mathcal{N}(\tilde{x}^{T}\mu_{\beta},\tilde{x}^{T}\Sigma_{\beta}\tilde{x})$$ 를 정규분포의 가법성을 활용해 더할 수 있다.

$$
\begin{align}
\tilde{y}\vert y &= \tilde{y}-\tilde{x}^{T}\beta+\tilde{x}^{T}\beta\vert y \nonumber \\
&\sim \mathcal{N}(\tilde{x}^{T}\mu_{\beta},\sigma^{2}+\tilde{x}^{T}\Sigma_{\beta}\tilde{x}) \nonumber
\end{align}
$$

즉, $$\tilde{y}$$의 Posterior Predictive Distribution은 새로운 독립변수 $$\tilde{x}$$에 $$\beta$$ Posterior Distribution의 평균값을 곱한 값을 평균으로 한다.

새롭게 구한 Posterior Predictive Distribution의 분산을 주목해볼 필요가 있다. 분산의 경우 2가지 항으로 구분되어 있는데 첫번째 항은 새로운 독립적인 표본의 노이즈를 표현한다. 두번째항은 $$\beta$$에 대한 불확실성을 반영한다. $$\beta$$에 대한 불확실성(분산)을 표현하는 $$\Sigma_{\beta} = \left(\frac{1}{\sigma^{2}}X^{T}X +\Sigma_{0}^{-1} \right)^{-1}$$ 식을 살펴보면 다음과 같은 이야기를 풀어갈 수 있다.

새로운 데이터가 추가될수록 $$X^{T}X$$ 값은 점점 커질 것이다. 그렇다면, 전체 $$\Sigma_{\beta}$$값은 작아질 것이다. 즉, 데이터가 추가될수록 $$\beta$$의 분산은 작아지는 것이다.

그렇다면, Posterior Predictive Distribution의 분산 term인 $$\sigma^{2}+\tilde{x}^{T}\Sigma_{\beta}\tilde{x}$$ 에서 2번째항은 데이터가 추가될수록 점점 미치는 영향력이 작아질 것이다. 그 결과 추가되는 데이터가 무한히 많아지면, $$N \to \infty$$, 이 Posterior Predictive Distribution의 분산은 $$\sigma^{2}$$의 부분만 남는다고 볼 수 있다.

결국 $$\sigma^{2}_{N}(x) \geq \sigma^{2}_{N+1}(x)$$라는 것인데 이에 대해서는 다음의 [논문](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.25.9575&rep=rep1&type=pdf)을 통해 증명됨을 확인할 수 있다.

![PPD](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/PPD_1.PNG?raw=true){: .align-center}{:width="70%" height="70%"}

이 그림은 간단한 베이지한 선형모델의 결과를 바탕으로 생성된 Posterior Predictive Distribution을 나타내는 Plot이다. 분석에 활용된 데이터 포인트에 대해서는 Plot 내에 점으로 표기하였다. 빨간색 선은 Posterior Mean으로 추정한 값을 활용해 모델을 적합시킨 것이고 파란색 선은 $$\beta$$의 Posterior Distribution을 통해 여러번 샘플링한 값을 가지고서 적합시킨 결과이다.

![PPD](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/PPD_2.PNG?raw=true){: .align-center}{:width="70%" height="70%"}

이 그림은 앞선 그림과 동일한 시행을 수행한 것이나 관측한 데이터 포인트 개수를 늘렸고 관측되는 데이터의 범위도 늘린 결과이다. 데이터 갯수를 늘린 결과, 우리는 Posterior Distribution을 더욱 정밀하게 추정할 수 있고 그에 따라 Posterior Predictive Distribution이 가질 수 있는 값의 범위가 좁혀지는 것을 확인할 수 있다. 범위가 좁다는 것은 분산이 작다는 것과 동등하다.

이 결과를 통해 우리는 2가지를 확인할 수 있다. Posterior Predictive Distribution의 불확실성은 데이터에 종속적이며, 데이터가 존재하는 주변에서 그 불확실성(분산)이 작다. 또한 불확실성의 정도는 관측된 데이터의 수가 증가함에 따라 감소하는 것을 확인할 수 있다.

이와 관련한 코드는 다음의 [링크](https://github.com/seolbluewings/pythoncode/blob/master/10.Posterior%20Predictive%20Distribution.ipynb) 에서 확인할 수 있습니다.




#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>
