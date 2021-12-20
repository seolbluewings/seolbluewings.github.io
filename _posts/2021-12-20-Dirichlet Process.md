---
layout: post
title:  "Dirichlet Process"
date: 2021-12-20
author: seolbluewings
categories: Statistics
---

[작성중...]

Dirichlet Process는 Dirichlet Distribution을 따르는 Random Process로 Unsupervised Learning에 자주 활용된다. 기존에 학습했던 Unsupervised Learning은 GMM, K-Means와 같은 Clustering 문제였다. 기존 GMM, K-Means 문제에서는 cluster의 개수 k를 분석가가 명시적으로 지정해야하는, 사람이 개입해야하는 이슈가 있었다. 이러한 문제에서 조금 더 자유로워지고자 할 때, cluster의 개수 k를 사람이 명시적으로 지정하지 않는 방식을 취하고자 할 때 선택할 수 있는 옵션이 Dirichlet Process라고 할 수 있다.

Dirichlet Process 이전에 Dirichlet Distribution과 Multinomial Distribution에 대해 짚고 넘어갈 필요가 있다.

#### Dirichlet Distribution & Multinomial Distribution

Multinomial Distribution은 어떠한 시행에서 M가지 값이 나올 수 있고 각각의 값이 나올 확률을 $$\theta_{1},...,\theta_{M}$$ 이라 할 때, N번의 시행에서 i번째 값이 $$x_{i}$$번 나타날 확률을 표현한다.

$$ p(\mathbf{x}\vert \mathbf{\theta}) = \frac{N!}{\prod_{i=1}^{M}x_{i}!}\prod_{i=1}^{M}\theta_{i}^{x_{i}} $$

한편, Dirichlet Distribution은 K차원의 continuous한 확률 변수를 return하는 확률 분포로 2이상의 자연수 K와 양수 $$\alpha_{1},...,\alpha_{K}$$, 양의 실수 $$\theta_{1},...,\theta_{K}$$ 가 $$\sum_{i=1}^{K}\theta_{i} = 1$$ 을 만족할 때, 다음과 같은 확률 분포를 갖는다.

$$ p(\theta \vert \alpha) = \frac{\Gamma\left(\sum_{i=1}^{K}\alpha_{i}\right)}{\prod_{i=1}^{K}\Gamma(\alpha_{i})} \prod_{i=1}^{K}\theta_{i}^{\alpha_{i}-1}  $$

Dirichlet Distribution으로 얻을 수 있는 결과 $$\theta = (\theta_{1},...,\theta_{K})$$ 가 Probability의 공리 조건들을 만족하기 때문에 우리는 Dirichlet Distribution의 확률 변수를 Multinomial Distribution의 parameter로 활용할 수 있다.

이 2개의 분포는 서로 Conjugate한 관계를 지니고 있는데 


포스팅 내용에 대한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/cheating%20sheet/pandas%20cheating%20sheet.ipynb)에서 확인 가능합니다.
