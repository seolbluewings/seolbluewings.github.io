layout: post
title:  "Bayesian Linear Regression"
date: 2019-04-22
author: YoungHwan Seol
categories: Bayesian
---

일반적으로 선형 회귀(linear regression)는 다음과 같이 표현된다.

$${\bf y} = {\bf X}{\bf \beta}+\epsilon $$

여기서 $$ \epsilson $$은 평균이 0, 분산이 $$\sigma^{2}$$ 인 가우시안 확률 변수이고 따라서 다음과 같이 적을 수 있다.

$$ {\it p}({\bf y}|{\bf X},{\bf \beta},\sigma^{2})= \mathcal{N}(y|{\bf X\beta},\sigma^{2}) $$


