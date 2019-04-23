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

입력 데이터 집합 $${\bf X}=\{{\bf X_{1}},...,{\bf X_{N}}\}$$ 과 타깃 변수 $${\bf y}=\{ y_1,...,y_{N}\}$$ 이 있다고 하자. 각 데이터가 먼저 언급한 분포로부터 독립적으로 추출되었다는 가정 하에 다음과 같은 Likelihood 함수를 얻을 수 있다.

$${\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2}) = \prod_{i=1}^{N} \mathcal{N}(y_{i}|{\bf x_{i}}{\bf \beta},\sigma^{2})$$

$$\log{\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2}) = \sum_{i=1}^{N}\log\mathcal{N}(y_{i}|{\bf x_{i}}{\bf \beta},\sigma^{2})$$

$$\log{\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2}) = -\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}({\bf y}-{\bf X\beta})^{T}({\bf y}-{\bf X\beta}) $$

이제 Maximum Likelihood 방법을 적용하면, $${\bf \beta}$$ 와 $$\sigma^{2}$$ 를 구할 수 있다. 



