---
layout: post
title:  "Gibbs Sampler"
date: 2019-05-22
author: YoungHwan Seol
categories: Bayesian
---

다음과 같은 모수가 존재한다고 하자. 

$$\mathbf{\theta}=(\theta_{1},\theta_{2},\theta_{3})$$ 

모수들의 결합 사후분포(joint posterior distribution)를 아는 것이 좋겠지만, 이 결합 사후분포

$$p(\theta_{1},\theta_{2},\theta_{3}|\mathbf{X})$$

가 계산하기 어려운 형태로 주어지는 반면, 완전 조건부 사후분포(full conditional posterior distribution)이 계산하기 쉬운 형태로 주어질 때가 있다. 

full conditional posterior distribution이란, 관심모수를 제외한 나머지가 모두 주어진 조건인 분포를 말하며 다음과 같다.

$$p(\theta_{1}|\theta_{2},\theta_{3},\mathbf{X})$$

$$p(\theta_{2}|\theta_{1},\theta_{3},\mathbf{X})$$ 

$$p(\theta_{3}|\theta_{1},\theta_{2},\mathbf{X})$$

Gibbs Sampler의 과정은 다음과 같다.

우선 $$\theta_{1},\theta_{2},\theta_{3}$$ 의 initial value인 $$\theta_{1}^{(0)},\theta_{2}^{(0)},\theta_{3}^{(0)}$$ 를 정한다.

첫번째 step은 다음과 같다.

$$\theta_{1}^{(1)} \sim p(\theta_{1}|\theta_{2}^{(0)},\theta_{3}^{(0)},|\bf{X})$$
$$\theta_{2}^{(1)} \sim p(\theta_{2}|\theta_{1}^{(1)},\theta_{3}^{(0)},|\bf{X})$$
$$\theta_{3}^{(1)} \sim p(\theta_{3}|\theta_{1}^{(1)},\theta_{2}^{(1)},|\bf{X})$$

두번째 step은 다음과 같다.

$$\theta_{1}^{(2)} \sim p(\theta_{1}|\theta_{2}^{(1)},\theta_{3}^{(1)},|\bf{X})$$
$$\theta_{2}^{(2)} \sim p(\theta_{2}|\theta_{1}^{(2)},\theta_{3}^{(1)},|\bf{X})$$
$$\theta_{3}^{(2)} \sim p(\theta_{3}|\theta_{1}^{(2)},\theta_{2}^{(2)},|\bf{X})$$

이처럼 여러 step을 반복하면 m번째 step은 다음과 같을 것이다.

$$\theta_{1}^{(m)} \sim p(\theta_{1}|\theta_{2}^{(m-1)},\theta_{3}^{(m-1)},|\bf{X})$$
$$\theta_{2}^{(m)} \sim p(\theta_{2}|\theta_{1}^{(m)},\theta_{3}^{(m-1)},|\bf{X})$$
$$\theta_{3}^{(m)} \sim p(\theta_{3}|\theta_{1}^{(m)},\theta_{2}^{(m)},|\bf{X})$$

이를 일반적인 방식으로 표현하면 다음과 같다. 관심모수가 다음과 같이 d 차원인 경우를 생각해보자.

$$\mathbf{\theta}=(\theta_{1},\theta_{2},\theta_{3},....,\theta_{d})$$

