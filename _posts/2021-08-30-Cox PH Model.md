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

설명변수 데이터 $$Z_{i}(t)$$를 포함한 생존 데이터는 $$(t_{i},\delta_{i},Z_{i}(t))$$ 로 표현될 수 있다. $$t_{i}$$ 와 $$\delta_{i}$$에 대해서는 이전 [포스팅](https://seolbluewings.github.io/statistics/2021/08/25/Survival-Analysis_1.html)을 참조하면 된다.

여기서 $$Z_{i}(t)$$ 는 i번째 데이터에 대한 설명변수 vector로 설명변수가 총 p개 있다고 하면 다음과 같이 표현할 수 있을 것이다. 그리고 만약 j번째 설명변수가 시간에 의존하지 않는 변수라면 $$z_{ij}(t) = z_{ij} $$ 로도 표현 가능하다.

$$ Z_{i}(t) = (z_{i1}(t),...,z_{ip}t)^{T} $$





#### 참조 문헌
1. R을 이용한 생존분석 기초
2. [생존 분석(Survival Analysis) 탐구 1편](https://velog.io/@jeromecheon/%EC%83%9D%EC%A1%B4-%EB%B6%84%EC%84%9D-Survival-Analysis-%ED%83%90%EA%B5%AC-1%ED%8E%B8)
3. [The basics of survival analysis](https://sakai.unc.edu/access/content/group/2842013b-58f5-4453-aa8d-3e01bacbfc3d/public/Ecol562_Spring2012/docs/lectures/lecture27.htm)

