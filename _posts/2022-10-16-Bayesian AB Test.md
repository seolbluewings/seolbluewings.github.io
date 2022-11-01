---
layout: post
title:  "Bayesian A/B Test"
date: 2022-10-16
author: seolbluewings
categories: Statistics
---

A/B 테스트는 온라인 서비스 적용, 대고객 마케팅 캠페인 등에서 자주 사용되는 개념으로, 서로 다른 A/B방법 중 어느 방법이 더 효과적인 방법인지 알아보기 위해 활용된다. A/B Test를 바탕으로 우리는 각 방식에 따라 관심지표(CTR 또는 반응률)가 어떠한 영향을 받는지 인과성을 유추해볼 수 있다.


![ABTEST](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/ab_test.jpg?raw=true){:width="70%" height="30%"}{: .aligncenter}


만약 동일한 마케팅 캠페인에서 A,B가 타겟 선정방식이 다르고 B의 반응률이 더 높다면, B의 타겟팅 방식이 더 낫다고 결론을 내리길 원한다. 그런데 B의 반응률이 A보다 얼마나 높아야 A보다 B가 낫다고 말할 수 있는지가 중요하다.

#### Frequentist의 방법

대중적으로 알려진 A/B Test 방식은 Frequentist의 가설검정이다. A와 B방식은 서로 차이가 없다고 가정하는 Null Hypothesis를 두고 이를 채택/기각하기 위해 p-value를 계산한다.

기존의 방식을 A라고 하고 새롭게 적용하는 방식을 B라고 한다면, Null Hypothesis는 A와 B의 차이는 존재하지 않는다고 보수적으로 접근한다. 이 과정에서 유의수준을 0.05로 설정하는데 유의수준이란 두 방법 사이의 차이가 없는데도 차이가 있다고 잘못 판단할 가능성(1종 오류, false positive)을 최대 5%까지만 제한하겠다는 것을 의미한다. (꼭 0.05로 할 필요는 없다)

가설검정의 결과 발생하는 p-value 값이 이 유의수준 0.05보다 작을 때, 우리는 Null Hypothesis를 기각한다. p-value는 두 방법론의 차이가 없다는 Null Hypothesis를 참으로 가정했을 때, 현재보다 더 극단적인 수치가 나올 확률을 의미한다. 이 p-value가 유의수준 0.05보다 작을 때, 우리는 Null Hypothesis를 기각한다.

이 방식은 Null Hypothesis에 좀 더 weight를 둔 가설 검정으로 이 Null Hypothesis가 잘못된 것으로 판단 내리기 위해서는 (기각을 위한) 충분한 양의 데이터가 필요하다. 또한 두가지 A,B 방식에 대한 차이의 유무를 판별할 수 있을 뿐이지 그 둘의 차이가 얼마나 발생하는지는 알기 어려운 이슈가 있다. p-value에 대한 이해가 어려운 것도 하나의 단점이라 할 수 있다.

#### Bayesian의 방법

반면, Bayesian A/B Test는 A와 B방법을 비교하여 B방법이 A보다 더 좋은 방법일 확률이 얼마나 되는지를 알려준다. 이 확률을 알기 위해서는 다음과 같은 사전 분포 가정과 데이터 처리가 필요하다.

A/B 테스트를 통해 비교할 항목이 반응률(또는 전환율)이라면, 데이터의 likelihood 함수로 Binomial Distribution을 활용하고 사전 분포로 Beta Distribution을 선택한다.

Binomial Distribution은 n번의 시도 중 성공한 횟수를 나타내는 분포인데 이는 n명의 마케팅 대상자 중 반응한 고객수에 대응할 수 있다. n명 중 x명이 마케팅에 반응했다하면, likelihood를 다음과 같이 표현할 수 있다.

$$
\begin{align}
x &\sim \text{Bin}(n,\pi) \nonumber \\
p(x) &= {n \choose x} \pi^{x}(1-\pi)^{1-x} \nonumber
\end{align}
$$

여기서의 $$\pi$$는 반응률로 우리의 관심 parameter라고 할 수 있다. 그렇다면, Bayesian 방식을 적용하기 위해서는 이 parameter $$\pi$$에 대한 Prior 분포를 설정해야 한다. 확률 $$\pi$$는 [0,1] 사이에서 정의되어야하므로 Binomial Distribution의 conjugate prior이면서 (0,1)사이에서 정의되는 Beta Distribution을 Prior로 설정한다.

$$
\begin{align}
\pi &\sim \text{Beta}(\alpha,\beta) \nonumber \\
p(\pi) &= \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)} \pi^{\alpha-1}(1-\pi)^{\beta-1} \quad \text{where} \; \pi \in (0,1) \nonumber
\end{align}
$$

Prior는 분석가의 주관에 따라 좌우되는 단점이 있다. 반응률 p에 대한 어떠한 사전지식도 없다고할 때, 단지 반응률이 (0,1) 사이에서 랜덤한 값을 가질 것이라는 무정보적인 사전 분포(non-informative prior)를 부여해야한다. Beta(1,1)의 경우 Uniform(0,1)과 동일하여 Beta(1,1)을 사용하면 된다.

이 둘을 활용하여 구할 수 있는 $$\pi$$에 대한 Posterior 분포는 Beta형태의 분포로 다음과 같다.

$$
p(\pi\vert x) \sim \text{Beta}(\alpha+x, \beta+n-x) \propto \pi^{x+\alpha-1}(1-\pi)^{n-x+\beta-1}
$$

A,B 각 방법론의 마케팅 시행고객 수(n)과 반응 고객수(x)를 활용하여 Beta Posterior 분포를 생성하고 각 Posterior 분포에서 난수를 생성하여 평균을 구하면 각각의 반응률 평균치를 구할 수 있다.

동일한 기준의 캠페인인데 방식A는 57,314명을 대상으로 시행되어 3,709명이 반응했고 방식B는 38,342명을 대상으로 시행되어 2,632명이 반응했다면 Bayesian A/B Test의 결과는 다음과 같을 것이다.

캠페인 A의 경우는 난수 추출이 $$ \text{Beta}(3710,53606) $$ 분포에서 시행되고 캠페인 B의 경우는 난수 추출이 $$ \text{Beta}(2633,35711) $$ 분포에서 시행된다. 약 10만번의 난수추출을 시행하여 각각의 histogram을 그려보면 다음과 같을 것이다.

캠페인 A가 B보다 나을 확률은 총 10만번의 난수추출 중에서 A에서 추출된 난수가 B에서 추출된 난수보다 큰 경우로 표현할 수 있다.

![ABTEST](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/ab_test2.png?raw=true){:width="70%" height="30%"}{: .aligncenter}

그림과 같이 histogram을 비교해보면 B의 방식이 A보다 좋다는 것을 시각적으로 확인 가능하다. A에서의 난수가 B에서 추출한 난수보다 더 클 확률은 0.83\%로 이는 A방식이 B방식보다 더 좋을 확률이 단 0.83\%에 불과하다는 의미한다. 압도적인 확률로 B가 낫다고 볼 수 있다.









Bayesian A/B Test에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/python_study/blob/master/01.study/bayesian_AB_test.py)에서 확인 가능하다.


#### 참조 문헌
1. [Bayesian AB Test](https://assaeunji.github.io/bayesian/2020-03-02-abtest/) <br>
2. [Bayesian A/B Testing with Expected Loss](https://miistillery.me/bayesian-ab-testing/)
