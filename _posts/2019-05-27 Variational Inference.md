---
layout: post
title:  "Variational Inference"
date: 2019-05-27
author: YoungHwan Seol
categories: Bayesian
---

Variational Inference란 복잡한 형태의 posterior 분포 $$p(z||x)$$를 다루기 쉬운 형태의 $$q(z)$$로 근사하는 것을 말한다.

다음과 같은 형태의 posterior distribution이 있다고 하자. 

$$p(z|x) = \frac{p(z,x)}{\int_{z}p(z,x)dz}$$ 

여기서 분모의 적분이 계산하기 어려운 경우 Variational Inference를 사용한다고 할 수 있다.

Variational Inference의 핵심적인 아이디어는 다음과 같다.

1. variational parameter $$\nu$$를 갖는 latent variables \{z_{1},z_{2},...,z_{m}\}의 분포$$q(z_{1},z_{2},...,z_{m}||\nu)$$를 찾는다.
2. 이 분포를 찾아가는 과정에서 posterior distribution에 가장 가까이 근사하는 모수 $$\nu$$를 찾아낸다.
3. 이렇게 구한 분포 $$q$$를 posterior 대신 사용한다.

그렇다면, 우리는 posterior distribution에 근사한 $$q(z)$$를 만들기 위해 쿨백-라이블러 발산(Kullback-Leibler Divergence)에 대해 이해해야 한다. 
