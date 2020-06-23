---
layout: post
title:  "가우시안 혼합 모델(Gaussian Mixture Model)"
date: 2020-06-12
author: YoungHwan Seol
categories: Statistics
---

가우시안 분포(정규 분포)를 선형으로 중첩시키는 경우, 우리는 단일 가우시안 분포를 사용할 때보다 더욱 다양한 종류의 모델을 생성해낼 수 있다. 혼합 가우시안 분포는 다음과 같이 가우시안 분포의 선형 결합 방식으로 표현할 수 있다.

$$
P(\mathbf{x}) = \sum_{k=1}^{K}\pi_{k} \mathcal{N}(\mathbf{x}\mid \mu_{k},\Sigma_{k})
$$

위와 같은 수식에 잠재 변수(latent variable) 개념을 도입하면 우리는 수식을 더욱 폭넓게 이해할 수 있다. K차원의 이산형 확률변수 $$z$$를 도입하자. 이 이산형 확률변수 $$\mathbf{z}$$는 $$z_{k} \in \{0,1\}$$ 및 $$\sum_{k}z_{k} = 1$$ 조건을 만족한다고 가정하자. 그렇다면, 확률변수 $$\mathbf{z}$$ 의 분포는 다음과 같이 $$p(z_{k}=1)=\pi_{k}$$ 로 표현할 수 있다. 여기서 $$\pi_{k}$$는 변수 $$z_{k}$$가 1이 되기 위한 확률일테고 $$\pi_{k}$$는 유효한 확률값이 되기 위해서 다음의 2가지 조건 $$ 0 \leq \pi_{k} \leq 1 $$, $$\sum_{k=1}^{K}\pi_{k}=1 $$ 을 만족시켜야 한다.

즉 확률변수 $$\mathbf{z}$$는 디리클레 분포를 따르며 $$ p(\mathbf{z}) = \sum_{k=1}^{K}\pi_{k}^{z_{k}}$$ 로 표현 가능하다.



##### 이 포스팅과 관련된 간단한 코드는 다음의 [주소](https://github.com/seolbluewings/code_example/blob/master/2.Cluster%20Analysis.ipynb)에서 확인할 수 있습니다.





