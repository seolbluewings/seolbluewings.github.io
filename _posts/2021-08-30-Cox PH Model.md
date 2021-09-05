---
layout: post
title:  "Cox Proportional Hazard Model"
date: 2021-08-30
author: seolbluewings
categories: Statistics
---

[작성중...]

생존 분석의 목표는 사건이 발생하기까지의 시간을 예측하고 더불어 생존 확률을 추정하는 것에 있다. 기존의 회귀, 분류 문제에서 사용하는 알고리즘을 생존 데이터에 바로 적용하기 어려운 것은 생존 데이터에 중도 절단이란 개념이 포함되어 있기 때문이다. 중도절단과 사건발생 시간을 고려하지 않는 모델은 잘못된 결론을 도출할 수 있다.

생존 분석에서 가장 대표적으로 사용하는 모델은 Cox 비례위험 모형(Cox Proportional Hazard Model)이다. Cox PH 모델은 위험함수(hazard function)이 설명변수에 대한 회귀식을 포함하는 모형으로 설명변수에 대해서는 parametric한 가정을 기저가 되는 위험함수에 대해서는 non-parametric한 가정을 하는 일종의 semi-parametric한 모형이다.

#### Cox PH Model

분석대상에 대한 위험함수가 설명변수에 의해 영향을 받는다면, 위험함수 $$h(t)$$는 설명변수 $$\mathbf{Z}$$에 도 함수의 변수로 사용해야할 것이다. Cox PH 모델의 가장 큰 특징이 바로 생존시간과 설명변수의 관계가 위험함수를 통해 표현된다는 것이다.

고객 이탈을 예측한다고 하면, 다음과 같은 설명 변수들이 사용될 수 있을 것이다.

- 무실적 일수, 카드발급채널, 카드연계 대출실행여부 등...

이러한 설명변수는 시간에 따라 값이 변화하는 변수일 수도, 변하지 않는 변수일 수도 있다.

설명변수 데이터 $$z_{i}(t)$$를 포함한 생존 데이터는 $$(t_{i},\delta_{i},z_{i}(t))$$ 로 표현될 수 있다. $$t_{i}$$ 와 $$\delta_{i}$$에 대해서는 이전 [포스팅](https://seolbluewings.github.io/statistics/2021/08/25/Survival-Analysis_1.html)을 참조하면 된다.

여기서 $$z_{i}(t)$$ 는 i번째 데이터에 대한 설명변수 vector로 설명변수가 총 p개 있다고 하면 다음과 같이 표현할 수 있을 것이다. 그리고 만약 j번째 설명변수가 시간에 의존하지 않는 변수라면 $$z_{ij}(t) = z_{ij} $$ 로도 표현 가능하다.

$$ z_{i}(t) = (z_{i1}(t),...,z_{ip}(t))^{T} $$

먼저 변수 $$z_{i}(t)$$가 시간 $$t$$에 의해 변하지 않는 변수(ex : 성별) 일 때만을 고려한 Cox PH Model을 고려해보자.

$$h(t\vert z_{i}) $$ 는 t시점에서 시간에 의존하지 않는 변수 $$z_{i}$$ 를 가진 대상에 대한 위험함수를 의미한다.

$$
\begin{align}
h(t\vert z_{i}) &= h_{0}(t)\text{exp}(z_{i}^{T}\beta) \nonumber \\
&= h_{0}(t)\text{exp}(z_{i1}\beta_{1}+....+z_{ip}\beta_{p}) \nonumber
\end{align}
$$

$$\beta = (\beta_{1},...,\beta_{p})$$ 는 회귀계수로 변수 $$z_{ik}$$ 가 1단위 증가할 때, 위험변수는 $$\text{exp}(\beta_{k})$$ 만큼 증가한다고 해석 가능하다.

Cox PH Model의 특징 중 하나는 로지스틱 회귀의 오즈비와 같이 개체간의 위험률의 비(hazard ratio)를 구할 수 있다는 것이다. 이 hazard ratio를 구하는 수식은 다음과 같다.

$$
\frac{h(t\vert z_{i})}{h(t\vert z_{j})} = \frac{ h_{0}(t)\text{exp}(z_{i}^{T}\beta) }{ h_{0}(t)\text{exp}(z_{j}^{T}\beta) } = \text{exp}\left(\sum_{k=1}^{p}(z_{ik}-z_{jk})\beta_{k}\right)
$$

hazard ratio의 수식을 통해서 확인할 수 있듯이 개체간의 위험비는 시간에 의존하지 않는 값이다. 오로지 변수의 값에 의존해 변하게 되는데 이는 시간에 의존하지 않는 변수들만 사용한 Cox PH Model의 한계로 다가올 수도 있다.

서비스를 6개월 전에 모바일 채널을 통해 가입한 사람의 이탈 가능성과 1개월 전에 동일한 채널을 통해 가입한 사람의 이탈 가능성이 동등한 것으로 간주하는 것이다. 이 둘의 이탈 가능성을 단순히 가입 채널이 같다는 이유만으로 동등하게 보는 것은 옳지 않다.

따라서 실질적으로 이 모델을 사용하기 위해선 Time dependent한 변수를 포함한 Cox PH Model에 대해서 알아야 한다.


#### 참조 문헌
1. R을 이용한 생존분석 기초
2. [생존 분석(Survival Analysis) 탐구 1편](https://velog.io/@jeromecheon/%EC%83%9D%EC%A1%B4-%EB%B6%84%EC%84%9D-Survival-Analysis-%ED%83%90%EA%B5%AC-1%ED%8E%B8)
3. [The basics of survival analysis](https://sakai.unc.edu/access/content/group/2842013b-58f5-4453-aa8d-3e01bacbfc3d/public/Ecol562_Spring2012/docs/lectures/lecture27.htm)

