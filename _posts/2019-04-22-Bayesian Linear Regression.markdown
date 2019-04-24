---
layout: post
title:  "Bayesian Linear Regression"
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

노이즈가 가우시안 분포를 따르면, 선형 모델에 대해 Maximum Likelihood 방식은 LSE 방식과 동일하다.

$$ \frac{\partial \log{\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2})}{\partial \beta}= -\frac{1}{2\sigma^{2}}({\bf y}-{\bf X\beta})^{T}({\bf y}-{\bf X\beta}) $$

$$ \beta = ({\bf X^{T}X})^{-1}{\bf X}{\bf y} $$

Likelihood를 최대화하는 방식으로 문제를 풀면 언제나 overfitting하는 복잡한 모델을 선택하게 되는데 베이지안 방법론으로 선형 회귀를 시행하면 Maximum Likelihood 방법에서 발생하는 overfitting 문제를 피할 수 있고, 훈련 데이터만 가지고 모델의 복잡도를 자동적으로 결정할 수 있다.

우선 $$\beta$$와 $$\sigma^{2}$$에 대한 prior distribution을 다음과 같이 $$\beta \sim \mathcal{N}(\beta_{0},\Sigma_{0})$$, $$\sigma^{2} \sim IG(a,b)$$ 로 설정한다.

prior distribution을 위와 같이 설정하면, posterior distribution은 다음과 같이 계산된다.

$${\it p}(\beta,\sigma^{2}|y) \propto (\sigma^{2})^{-n/2} \cdot exp\{ \frac{1}{2\sigma^{2}}(y-X\beta)^{T}(y-X\beta)\} \times exp\{ -\frac{1}{2}(\beta-\beta_0)^{T}\Sigma^{-1}_{0}(\beta-\beta_{0})\} \times (\sigma^{2})^{-a-1}\cdot exp\{-b/\sigma^{2}\} $$

$${\it p}(\beta|\sigma^{2},y) \propto exp \bigg[-\frac{1}{2}\bigg\{\frac{1}{\sigma^2}\beta^{T}X^{T}X\beta-\frac{2}{\sigma^2}\beta^{T}X^{T}y \bigg\}-\frac{1}{2}\bigg\{\beta^{T}\Sigma^{-1}_{0}\beta-2\beta^{T}\Sigma^{-1}_{0}\beta_{0}\bigg}\bigg]$$

$${\it p}(\beta|\sigma^{2},y) \propto exp \bigg[ -\frac{1}{2}\bigg{ \beta^{T}\bigg(\frac{1}{\sigma^2}X^{T}X+\Sigma^{-1}_{0}\bigg)\beta -2\beta^{T}\bigg(\frac{1}{\sigma^2}X^{T}y+\Sigma^{-1}_{0}\beta_{0}\bigg)\bigg}\bigg] $$

$${\it p}(\beta|\sigma^{2}|y) \sim \mathcal{N}(\mu_{\beta},\Sigma_{\mu})$$

여기서 $$\Sigma_{\beta}=\bigg( \frac{1}{\sigma^2}X^{T}X+\Sigma^{-1}_{0}\bigg)^{-1}$$ 이며 $$ \mu_{\beta}=Sigma_{\beta}^{-1}\cdot\bigg(\frac{1}{\sigma^2}X^{T}y+\Sigma^{-1}_{0}\beta_{0}\bigg)$$ 이다.