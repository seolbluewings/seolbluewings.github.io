---
layout: post
title:  "Survival Analysis (1)"
date: 2021-08-25
author: seolbluewings
categories: Statistics
---

[작성중...]

고객 이탈예측 모형 개발을 위해 생존분석(Survival Analysis)관련 자료를 정리하고자 포스팅을 작성한다. 생존분석은 어떠한 사건이 발생할 확률을 시간이라는 변수와 같이 고려하는 분석 방법으로 의료 통계에서 많이 사용되는데 이 방법을 고객 이탈을 예측하는 모델을 개발하는 과정에서도 사용한다.

생존분석을 공부하기 위해 평소 통계학 문헌에서 언급되지 않았던 몇가지 생소한 개념에 대한 사전 정의가 필요하고 이번 포스팅은 그러한 개념들을 정리하기 위한 용도로 작성할 것이다. 생존분석 기법을 고객 이탈예측 시 적용하기 위해 공부하는 것이므로 몇가지 단어 선택도 변경하여 정리하기로 한다.

#### 시간(time)

생존분석은 고객이 서비스를 가입(또는 구매)한 시점으로부터 이탈(사건이 발생)할 때까지의 시간 구간(time interval) 데이터에 관심이 있다. 그래서 우리의 관심 대상이 될 반응 변수는 고객이 이탈하는 사건이 발생할 때까지 걸리는 시간이다.

생존분석 실생 시, 시간 경과에 따른 고객의 잔존 확률을 구하게 된다. 이 때 시간은 하나의 독립 변수로 사용된다. 대신 이 때 시간은 상대적 시간 개념으로 분석 대상이 관측되기 시작한 시점(즉, 서비스를 가입한 시점)을 0으로 하여 시간을 계산하게 된다.

#### 사건(event)

잔존의 반대인 이탈을 의미한다. 고객이 이탈하는 것은 한번 발생하며 0과 1로 구분 가능하다.

#### 생존함수(survival function)

생존함수는 관찰대상이 특정 기준 시간보다 더 늦게 이탈하거나 이탈 자체가 일어나지 않을 확률을 계산하는 함수다. 즉 사건이 발생하기까지의 시간에 대한 함수로 다음과 같이 함수 $$S(t)$$로 표현 한다.

$$ S(t) = p(T > t) = \int_{t}^{infty}f(x)dx = 1- p(T \leq t) = 1- F(t) $$

생존함수는 확률을 계산하는 함수이므로 $$ 0 \leq S(t) \leq 1$$ 조건을 만족한다. 그리고 이탈이 관측되는 시간 t는 관측 즉시 발생할 수도 있고 또는 영영 발생하지 않을 수도 있다. 따라서 이 함수에서 t의 범위는 $$ t \in [0,\infty) $$ 이며, $$S(0) = 1, S(\infty) = 0$$ 인 것도 쉽게 받아들일 수 있다. 더불어 생존함수가 non-increasing function인 것도 받아들일 수 있다.

그래서 $$p(T>t)$$ 는 고객의 이탈이 t시점 이후 발생할 확률이며 이는 적어도 t시점까지는 고객이 이탈하지 않고 서비스를 이용한다고 해석할 수 있다.

#### 위험함수 (hazard function)

조건부 확률에 대한 함수 $$h(t)$$ 이며, t시점에 잔존했다는 조건 하, t시점에 이탈이 발생할 확률을 계산한다. 즉 t시점 이전까지는 이탈이 발생하지 않았는데 특정한 시점 t에 이탈이 발생할 확률을 의미한다.

$$
\begin{align}
h(t) &= \text{lim}_{\Delta \to 0} \frac{ p(t \leq T \leq T + \Delta t\vert T \geq t) }{ \Delta t} = \frac{f(t)}{S(t)} = \frac{ \frac{d}{dt}F(t) }{S(t)} \nonumber \\
&= \frac{ \frac{d}{dt}(1-S(t))}{S(t)} = \frac{-S'(t) }{S(t)} = -d\log{S(t)}
\end{align}
$$

위험함수는 증가/감소/상수형태 모두 가능하다.

만약 서비스 가입 기간 후 오랜 시간이 지나 평균적인 서비스 이용 기간을 넘어섰다면 이탈 위험은 증가할 것이다.

반대로 서비스 가입 초기는 한번 사용해보고 즉시 이탈할 수 있는 가능성이 있다. 그러나 일정기간 후 서비스 이용에 대한 안정기에 접어들면 위험함수는 (일시적으로라도) 감소가 가능하다.

#### 누적위험함수(cumulative hazard function)

시간 0에서부터 t시점까지의 위험함수를 적분한 값이며 이는 t시점까지 이탈이 발생할 확률을 모두 더한 것이다.

$$ H(t) = \int_{0}^{t}h(u)du = \int_{0}^{t}\frac{ \frac{d}{du}(1-S(u)) }{ S(u) } = -\log{S(t)} $$

수식을 통해 알 수 있는 것처럼 $$S(t) = \text{exp}\left(- H(t)\right)$$ 이다.

#### 중도절단(censoring / censored)




상기 내용에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Latent%20Dirichlet%20Allocation.ipynb)에서 확인 가능합니다.


#### 참조 문헌
1. [Gibbs Sampling for LDA](https://www.edwith.org/machinelearning2__17/lecture/10883?isDesc=false)
2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
3. [Collapsed Gibbs Sampler for LDA](https://lee-jaejoon.github.io/stat-collapsedGibbs/)
4. [토픽 모델링, LDA Python 코드](https://donghwa-kim.github.io/lda_code.html)
