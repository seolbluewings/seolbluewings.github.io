---
layout: post
title:  "EM 알고리즘(EM Algorithm)"
date: 2020-06-27
author: YoungHwan Seol
categories: Statistics
---

EM알고리즘은 잠재변수(latent variable)를 갖는 확률 모델의 MLE 값을 찾는 과정에서 활용되는 기법이다. 잠재변수를 활용하는 가우시안 혼합 모델에 관한 추정에 자주 활용되는 기법이기도 하다.

우선 관측변수 $$\mathbf{X}=\{x_{1},...x_{N}\}$$ 이라 하자. 잠재변수 $$\mathbf{Z}=\{z_{1},z_{2},...z_{n} }$$ 는 이 논의를 진행하는 과정에서 이산형 변수라고 가정하자. 만약 $$\mathbf{Z}$$가 연속형이라면, 아래의 과정에서 합표기가 되어있는 것을 적분으로 바꾸면 된다. 그리고 모델에 활용되는 변수들을 $$\Theta$$ 라고 표현하자.

우리의 목표는 $$P(\mathbf{X} \mid \Theta)$$ 혹은 $$l(\Theta \mid \mathbf{X})$$ 를 최대화시키는 $$\Theta$$ 값을 구하는 것이다. 이 $$P(\mathbf{X} \mid \Theta)$$에 대한 log likelihood는 다음과 같이 표현할 수 있다.

$$
\text{ln}P(\mathbf{X}\mid\Theta) = \text{ln}\{\sum_{z}P(\mathbf{X},\mathbf{Z}\mid\Theta)\}
$$

$$\mathbf{X}$$ 의 관측값에 따른 잠재변수 $$\mathbf{Z}$$를 알게 되었다고 한다면, $$\{\mathbf{X},\mathbf{Z}\}$$ 를 완전한(complete) 데이터 집합이라 표현할 수 있다. $$\mathbf{Z}$$는 기존에 알지 못했던 것이기 때문에 missing value, $$\mathbf{X}$$는 기존에 알고 있었기 때문에 observed value라고 할 수 있다.





