---
layout: post
title:  "Markov Chain Theory"
date: 2019-03-22 
author: YoungHwan Seol
categories: Bayesian
---

유한의 discrete state space $${\it X}$$ 에 대해, 다음과 같은 확률과정(sotchastic process)가 있다고 가정하자.

$$\\{x^{(t)},t=0.1,2,...\\}$$

예를 들자면, $$x^{t}$$는 t번째 날짜의 날씨라고 할 수 있다.

만약 이 날씨의 확률과정이 다음과 같은 식을 만족한다면, 우리는 이를 Markov Chain이라 부를 수 있다.

$$ p(x^{(t+1)}|x^{(0)},...,x^{(t-1)},x^{(t)})=p(x^{(t+1)}|x^{(t)}) $$

다르게 표현하자면,  $$x^{(t+1)}$$ 은 $$x^{(t)}$$가 given일 때, $$ \\{ x^{(0)},...,x^{(t-1)} \\} $$ 과 독립이라 할 수 있고 즉, (t+1)번째 날의 날씨는 t번째 날의 날씨에만 영향을 받으며 그 이전의 날씨에는 영향을 받지 않는다고 볼 수 있다.

Markov Chain 과정은 다음과 같은 특징을 가지고 있다.

 





