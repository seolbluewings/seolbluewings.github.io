---
layout: post
title:  "Bayesian A/B Test"
date: 2022-10-16
author: seolbluewings
categories: Statistics
---

[작성중...]

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
x &\sim \text{Bin}(n,p) \nonumber \\
p(x) = {n \choose x} p^{x}(1-p)^{1-x} \nonumber
\end{align}
$$






신세계 L\&B의 와인 정보를 Crawling하는 코드는 다음의 [링크](https://github.com/seolbluewings/python_study/blob/master/01.study/web_crawling.py)에서 확인 가능하다.


#### 참조 문헌
1. [파이썬을 이용한 웹 스크래핑](https://www.boostcourse.org/cs201/joinLectures/179628) <br>
2. 데이터분석 실무 with 파이썬
