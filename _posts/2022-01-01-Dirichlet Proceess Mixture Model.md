---
layout: post
title:  "Dirichlet Process Mixture Model"
date: 2022-01-01
author: seolbluewings
categories: Statistics
---

[작성중...]

Mixture Model에서 Dirichlet Distribution을 사용하는 일반적인 방식은 parameter의 차원이 k로 고정되어 있고 이 k차원의 parameter에 대한 prior로 활용하는 것이었다. GMM에서 cluster의 개수는 정해져 있었고 k번째 cluster로 할당될 latent variable $$z_{k}$$를 정의하고 $$p(z_{k}=1) = \pi_{k}$$ 에서의 $$\pi_{k}$$에 대한 prior로 Dirichlet Distribution을 활용했다.

그러나 Dirichlet Process Mixture Model(DPMM)은 k를 특정 차원으로 한정짓지 않고 $$k \to \infty$$ 인 경우에 대해 논의하며 $$k \to \infty$$ 처리로 인해 DP를 활용하게 된다.

![DPMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/DP4.PNG?raw=true){:width="70%" height="70%"}{: .aligncenter}

DPMM에 대한 Graphical View는 위의 그림과 같다. 기존 H로 표현되었던 Base Distribution $$G_{0}$$ 와 $$\alpha$$가 DP로 인한 Multinomial Distribution $$G$$를 생성하며 데이터에 대한 분포는 $$G$$에서 결정된 $$\theta_{i}$$로 인해 결정된다.

이 Graphical View는 Chinese Restaurant Process를 통해 보다 직관적으로 다가온다.

![DPMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/DP3.png?raw=true){:width="70%" height="70%"}{: .aligncenter}



#### 참고문헌

1. [인공지능 및 기계학습심화](https://www.edwith.org/aiml-adv/joinLectures/14705)
2. [Density Estimation with Dirichlet Process Mixtures using PyMC3](https://austinrochford.com/posts/2016-02-25-density-estimation-dpm.html)