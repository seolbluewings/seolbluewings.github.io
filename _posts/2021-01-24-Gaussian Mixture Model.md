---
layout: post
title:  "Gaussian Mixture Model"
date: 2021-01-24
author: seolbluewings
categories: Bayesian
---

[작성중...]

우리는 통계문제를 해결하는 과정에서 데이터가 가우시안 분포(정규 분포)를 따를 것이라는 가정을 자주 한다. 그러나 데이터의 분포를 단 1개의 가우시안 분포만을 사용하여 표현하려는 것은 위험한 부분이 있다. 다음과 같은 경우를 고려해보자.

![GMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/GMM_1.png?raw=true){:width="60%" height="60%"}

그림을 통해 확인할 수 있는 것처럼 이 데이터들은 2개의 집단으로 나누어진 것으로 판단하는 것이 옳다. 하나의 가우시안 분포만으로는 데이터를 설명하기는 부적절하다. 대신 2개의 가우시안 분포를 선형 결합시킨다면, 우리는 이 데이터 집합을 더욱 잘 표현할 수 있다.

가우시안 분포들을 선형 결합하여 우리는 새로운 확률 분포를 생성해낼 수 있다.

![GMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/GMM_2.png?raw=true){:width="60%" height="60%"}

이 이미지는 파란색 곡선으로 표기된 3개의 가우시안 분포를 혼합하여 새로운 1개의 분포(빨간색)를 생성해낸 것을 의미한다. 새로운 분포를 생성하는 과정에서 우리는 혼합하는 가우시안 분포의 개수를 조정할 수도 있고 선형 결합의 대상이 되는 가우시안 분포의 parameter 값을 조정할 수도 있다. 또한 선형 결합의 계수를 조정함으로써 우리는 새롭게 생성되는 분포를 변주시킬 수 있다.

가우시안 혼합 분포(Gaussian Mixture Model)의 일반식은 다음과 같다.

$$ p(x) = \sum_{k=1}^{K} \pi_{k}\mathcal{N}(x\vert \mu_{k},\Sigma_{k})$$

이 수식은 총 K개의 가우시안 분포를 선형 결합하여 새로운 분포를 생성해내는 것을 표현한다. 또한 선형 결합의 대상이 되는 k번째 가우시안 분포의 평균과 분산이 각 $$\mu_{k},\Sigma_{k}$$임을 나타낸다.

이 수식에서 parameter $$\pi_{k}$$는 mixing coefficient로 불리며 $$\sum_{k=1}^{K}\pi_{k}=1$$ 조건을 만족한다. 또한 $$\mathcal{N}(x\vert \mu_{k},\Sigma_{k})$$이기 때문에 모든 k에 대하여 $$\pi_{k} \geq 0$$인 것은 $$p(x) \geq 0$$ 을 만족시키기 위한 충분조건이다. 그리고 모든 것을 종합해보면 우리는 $$ 0 \leq \pi_{k} \leq 1$$ 임을 알 수 있고 mixing coefficient $$\pi_{k}$$가 확률의 조건을 만족시키는걸 알 수 있다.

$$\pi_{k} = p(k)$$ 로 k번째 가우시안 분포에 속할 사전 확률(Prior)로 간주할 수 있다. $$\mathcal{N}(x\vert \mu_{k},\Sigma_{k})$$ 는 k가 주어진 상황에서 x의 확률, $$p(x \vert k)$$로 볼 수 있다.

$$p(x) = \sum_{k=1}^{K}p(k)p(x\vert k) = \sum_{k=1}^{K}\pi_{k}\mathcal{N)(x\vert \mu_{k},\Sigma_{k})$$

가 성립되어 Posterior distribution에 해당하는 $$p(k\vert x)$$ 값은 다음과 같이 계산될 것이다.

$$
p(k\vert x) = \frac{p(k)p(x\vert k)}{\sum_{k=1}^{K}p(x)p(x\vert k)} = \frac{\pi_{k}\mathcal{N}(x \vert \mu_{k},\Sigma_{k})}{\sum_{k=1}^{K}\pi_{k}\mathcal{N}(x\vert \mu_{k},\Sigma_{k})}
$$



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
