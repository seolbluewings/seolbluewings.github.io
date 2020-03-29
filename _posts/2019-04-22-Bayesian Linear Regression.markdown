---
layout: post
title:  "베이지안 선형 회귀(Bayesian Linear Regression)"
date: 2019-04-22
author: YoungHwan Seol
categories: Bayesian
---

일반적으로 선형 회귀(linear regression)는 다음과 같이 표현된다.

$${\bf y} = {\bf X} {\bf \beta}+\epsilon $$

여기서 $$ \epsilon $$은 평균이 0, 분산이 $$\sigma^{2}$$ 인 가우시안 확률 변수이고 따라서 다음과 같이 적을 수 있다.

$$ {\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2}) \sim \mathcal{N}(y|{\bf X\beta},\sigma^{2}) $$

입력 데이터 집합 $${\bf X}=\{ {\bf X_{1}},...,{\bf X_{N}} \}$$ 과 타깃 변수 $${\bf y}=\{ y_1,...,y_{N}\}$$ 이 있다고 하자. 각 데이터가 먼저 언급한 분포로부터 독립적으로 추출되었다는 가정 하에 다음과 같은 Likelihood 함수를 얻을 수 있다.

$${\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2}) = \prod_{i=1}^{N} \mathcal{N}(y_{i}|{\bf x_{i}}{\bf \beta},\sigma^{2})$$

$$\log{\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2}) = \sum_{i=1}^{N}\log\mathcal{N}(y_{i}|{\bf x_{i}}{\bf \beta},\sigma^{2})$$

$$\log{\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2}) = -\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}({\bf y}-{\bf X\beta})^{T}({\bf y}-{\bf X\beta}) $$

노이즈가 가우시안 분포를 따르면, 선형 모델에 대해 Maximum Likelihood를 최대화하는 방식은 LSE 방식과 동일하다.

$$ \frac{\partial \log{\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2})}{\partial \beta}= -\frac{1}{2\sigma^{2}}({\bf y}-{\bf X\beta})^{T}({\bf y}-{\bf X\beta}) $$

$$ \beta = ({\bf X^{T}X})^{-1}{\bf X}{\bf y} $$

Likelihood를 최대화하는 방식으로 문제를 풀면 언제나 overfitting하는 복잡한 모델을 선택하게 되는데 베이지안 방법론으로 선형 회귀를 시행하면 Maximum Likelihood 방법에서 발생하는 overfitting 문제를 피할 수 있고, 훈련 데이터만 가지고 모델의 복잡도를 자동적으로 결정할 수 있다.

우선 $$\beta$$와 $$\sigma^{2}$$에 대한 prior distribution을 다음과 같이 $$\beta \sim \mathcal{N}(\beta_{0},\Sigma_{0})$$, $$\sigma^{2} \sim IG(a,b)$$ 로 설정한다.

prior distribution을 위와 같이 설정하면, posterior distribution은 다음과 같이 계산된다.

$${\it p}(\beta,\sigma^{2}|y) \propto (\sigma^{2})^{-n/2} \cdot exp\{ \frac{1}{2\sigma^{2}}(y-X\beta)^{T}(y-X\beta)\} \times exp\{ -\frac{1}{2}(\beta-\beta_0)^{T}\Sigma^{-1}_{0}(\beta-\beta_{0})\} \times (\sigma^{2})^{-a-1}\cdot exp\{-b/\sigma^{2}\} $$

$${\it p}(\beta|\sigma^{2},y)\propto exp[-\frac{1}{2}\{\frac{1}{\sigma^{2}}\beta^{T}X^{T}X\beta-\frac{2}{\sigma^{2}}\beta^{T}X^{T}y\}-\frac{1}{2}\{\beta^{T}\Sigma^{-1}_{0}\beta-2\beta^{T}\Sigma^{-1}_{0}\beta_{0}\}]$$

$${\it p}(\beta|\sigma^{2},y)\propto exp[-\frac{1}{2}\{\beta^{T}(\frac{1}{\sigma^{2}}X^{T}X+\Sigma^{-1}_{0})\beta -2\beta^{T}(\frac{1}{\sigma^{2}}X^{T}y+\Sigma^{-1}_{0}\beta_{0})\}]$$

$${\it p}(\beta|\sigma^{2},y) \sim \mathcal{N}(\mu_{\beta},\Sigma_{\beta})$$

여기서 $$\Sigma_{\beta}=\bigg( \frac{1}{\sigma^2}X^{T}X+\Sigma^{-1}_{0}\bigg)^{-1}$$ 이며 $$\mu_{\beta}=\Sigma_{\beta}^{-1}\cdot\bigg(\frac{1}{\sigma^2}X^{T}y+\Sigma^{-1}_{0}\beta_{0}\bigg)$$ 이다.

$$\sigma^{2}$$에 대한 posterior distribution을 구하면,

$${\it p}(\sigma^{2}|\beta,y) \propto (\sigma^{2})^{-n/2}\times(\sigma^{2})^{-a-1}\cdot exp[-\frac{1}{\sigma^{2}}\{ \frac{1}{2}(y-X\beta)^{T}(y-X\beta)+b\}]$$

$${\it p}(\sigma^{2}|\beta,y) \sim IG(\frac{n}{2}+a,\frac{1}{2}(y-X\beta)^{T}(y-X\beta)+b)$$

과 같이 구할 수 있다.

![Bayesian_Linear_Regression](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Figure3.7.png?raw=true)

위의 그림은 1차원 데이터 x에 대한 회귀모델을 나타낸 것으로 $$y=\beta_{0}+\beta_{1}x$$ 형태이며 이 모델 상에서 데이터 집합의 크기가 커짐에 따른 베이지안 학습 결과를 보여준다. 현재의 posterior 분포는 새로운 데이터 포인트가 관측된 후, 새로운 prior distribution이 되는 베이지안 학습의 순차적인 모습을 보여준다.

1행은 데이터를 하나도 관측하지 못한 상황. 가운데 열에 있는 prior distribution 공간에 있는 $${\bf \beta}$$ 를 추출하여, 마지막열에 6개의 모델을 그려냈다.

2행은 첫번째 데이터(파란 원)를 관측한 이후의 상황. 가장 왼쪽열은 Likelihood $${\it p}(y\|x,\beta)$$ 를 $$\beta$$ 에 대한 함수로 그린 것이다. 이 Likelihood를 1행의 2번째열인 prior distribution과 곱하면, 2행의 2번째 열값인 posterior distribution을 구할 수 있다. 그리고 이 posterior distribution에서 추출한 $$\beta$$를 바탕으로 마지막열에 모델을 그려낸다.

마찬가지로 3번째 행은 두번째 데이터를 관측한 이후의 상황이며, 두번째 데이터에 대한 Likelihood와 이전의 posterior가 prior의 역할을 하여 3행의 2번째 열에 있는 $$\beta$$의 posterior 분포를 얻는다. 이 posterior로부터 추출한 $$\beta$$를 바탕으로 한 모델이 3행 3열에 위치해있다.

이와같이 베이지안 방식으로 업데이트 모델을 만들어낼 수 있다. sample이 하나 추가 될 때, 기존의 posterior distribution은 prior distribution으로 활용될 수 있으며 sample이 추가 될수록 posterior 분포가 특정한 값에 가까워진다.

앞서 우리는 2가지 conditional posterior를 다음과 같이 구했다.
$${\it p}(\beta|\sigma^{2},y) \sim \mathcal{N}(\mu_{\beta},\Sigma_{\beta})$$
$${\it p}(\sigma^{2}|\beta,y) \sim IG(\frac{n}{2}+a,\frac{1}{2}(y-X\beta)^{T}(y-X\beta)+b)$$

다음의 Gibbs Sampling 과정을 통해 $$\beta, \sigma^{2}$$에 대한 근사적인 추론이 가능하다.

k-1번째 step에서 $$(\beta^{(k-1)},\sigma^{2(k-1)})$$ 값이 주어지면, k번째 step은 다음과 같을 것이다.
1. $$\beta^{(k)}$$에 대한 추출은 $$\Sigma_{\beta}=\bigg(\frac{1}{\sigma^{2(k-1)}}X^{T}X+\Sigma^{-1}_{0}\bigg)^{-1}$$, $$\mu_{\beta}=\Sigma_{\beta}^{-1}\cdot\bigg(\frac{1}{\sigma^{2(k-1)}}X^{T}y+\Sigma^{-1}_{0}\beta_{0}\bigg)$$ 이고 $$\beta^{(k)} \sim \mathcal{N}(\mu_{\beta},\Sigma_{\beta})$$ 의 과정을 따를 것이다.
2. $$\sigma^{2(k)}$$에 대한 추출은 앞서 과정에서 update한 $$\beta^{(k)}$$를 바탕으로 $$(y-X\beta^{(k)})^{T}(y-X\beta^{(k)})$$를 구하고 $$\sigma^{2(k)} \sim IG(\frac{n}{2}+a, \frac{1}{2}(y-X\beta^{(k)})^{T}(y-X\beta^{(k)})+b)$$ 의 과정을 통해 $$\sigma^{2(k)}$$를 구한다.

$$\beta$$에 대한 prior를 $$\beta_{0}={\hat \beta_{LSE}}$$, $$\Sigma_{0}=c\sigma^{2}({\bf X}^{T(}{\bf X})^{2}$$ 라 놓으면, (즉 $$\Sigma_{0}$$를 $${\hat \beta_{LSE}}$$의 분산 $$\sigma^{2}({\bf X}^{T}{\bf X})^{2}$$ 에 비례하도록 설정하면) 다음과 같은 posterior를 구할 수 있다.

$$ \Sigma_{\beta}=\frac{c}{c+1}\sigma^{2}({\bf X}^{T}{\bf X})^{-1} $$ , $$ \mu_{\beta} = {\hat \beta_{LSE}} $$

$$p(\beta|\sigma^{2},y) \sim \mathcal{N}(\mu_{\beta},\Sigma_{\beta})$$

$$p(\sigma^{2}|y) \sim IG(\frac{n}{2}+a,\frac{1}{2}y^{T}(I-H)y+b)$$

이 경우는 앞선 경우와 달리 Gibbs sampling이 아닌 직접 $$\beta$$와 $$\sigma^{2}$$의 posterior 표본들을 다음과 같은 과정을 통해 구할 수 있다.

1. $$\sigma^{2(k)}$$ 추출 : $$\sigma^{2(k)} \sim IG(\frac{n}{2}+a,\frac{1}{2}y^{T}(I-H)y+b)$$

2. $$\beta^{(k)}$$ 추출 : $$\beta^{k}\|\sigma^{2(k)} \sim \mathcal{N}({\hat \beta},\frac{c}{c+1}\sigma^{2(k)}({\bf X}^{T}{\bf X})^{-1})$$

$$\beta$$ 를 알아내는 것보다는 새로운 $${\bf x}$$에 대하여 y의 값을 예측하는 것이 더 중요할 수 있다. 새로운 독립적인 데이터 포인트 $${\tilde x}$$가 주어졌을 때, 이에 대응하는 예측된 $${\tilde y}$$에 대한 예측 분포(predictive distribution)은 다음과 같을 것이다. 

$$p({\tilde y}|y,{\tilde x},X,\sigma^{2},\Sigma_{0})=\int p({\tilde y}|\beta,{\tilde x},\sigma^{2}) p(\beta|y,X,\sigma^{2},\Sigma_{0})d\beta $$

앞서 $$\mu_{\beta}=\Sigma_{\beta}^{-1}\cdot\bigg(\frac{1}{\sigma^2}X^{T}y+\Sigma^{-1}_{0}\beta_{0}\bigg)$$ 와 $$\Sigma_{\beta}=\bigg( \frac{1}{\sigma^2}X^{T}X+\Sigma^{-1}_{0}\bigg)^{-1}$$ 를 구했고

$$ {\tilde y}|y \sim \mathcal{N}({\tilde x}^{T}\mu_{\beta},\sigma^{2}_{n}({\tilde x}))$$

$$\sigma^{2}_{n}({\tilde x})=\sigma^{2}+{\tilde x}^{T}\Sigma_{\beta}{\tilde x}$$

이처럼 예측 분포의 분산은 2가지 항으로 구성되어 있고 첫번째 항은 데이터의 노이즈이며 두번째 항은 $$\beta$$에 의 posterior variance로 매개변수 $$\beta$$에 대한 불확실성을 표현한다. 이 둘은 각 독립적인 가우시안 분포이므로 분산을 합할 수 있고 추가적인 데이터 포인트들이 관측되면, posterior distribution은 더 좁아질 것이다. $$\sigma^{2}_{N+1}(x) \leq \sigma^{2}_{N}(x)$$








