---
layout: post
title:  "Partial Dependence Plot"
date: 2021-05-09
author: seolbluewings
categories: Statistics
---

(작성중...)

모델은 보통 2가지 관점에서 평가를 받는다. 관점 하나는 설명력이고 또 다른 관점 하나는 예측력이다. 보통 예측력이 좋은 모델일수록 해석력이 떨어지는 것으로 알려져 있다. 여기서 모델을 해석한다는 것은 변수 $$\mathbf{X}$$가 1단위 증가할 때, 반응변수 $$\mathbf{y}$$가 얼마나 변하는가? 를 알아내는 것을 의미한다. 결국 이는 Input에 의한 Output의 변동성을 체크한다는 것이다.

그러나 앙상블 모형에서부터 딥러닝까지 모델이 복잡해질수록 예측력은 높아지나 Input의 변동이 Output 변동에 얼마나 영향을 미쳤는지를 파악하기가 점점 어려워진다. 이렇게 예측력이 높으나 해석력이 떨어지는 모델을 Black Box 모델이라 부른다.

회귀분석의 경우, 회귀계수 $$\beta$$를 통해 Input 변화에 대한 Output 변화의 수준, 방향성을 파악할 수 있다. 이 방식은 Input $$\mathbf{X}$$가 1단위 증가할 때, $$\mathbf{y}$$가 얼마나 변하였는가를 파악하는 것이며 굉장히 직관적인 해석이 가능하다. 또한 우리는 어떤 Input 변수가 Output에 가장 영향을 많이 주었는지 파악할 수 있고 결과적으로 Input 변수의 상대적 중요도(Feature Importance)를 파악할 수 있다.

Black Box 모델에서도 Input 변수의 변동에 따른 Output의 변화를 살펴보고 싶은 욕구가 생기기 마련인데 이러한 이슈를 해결하기 위해 제안된 방법이 바로 Partial Dependence Plot(PDP) 이다.

#### Concept of PDP(Partial Dependence Plot)

PDP는 명칭을 통해서 확인할 수 있듯이, Plot을 그리는 방법이다.


#### 참조 문헌
1. [위키백과 오픈 API](https://ko.wikipedia.org/wiki/%EC%98%A4%ED%94%88_API) <br>
2. [직장인을 위한 데이터분석 실무 with 파이썬](https://wikibook.co.kr/playwithdata/)
