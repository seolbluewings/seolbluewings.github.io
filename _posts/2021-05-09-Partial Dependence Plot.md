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

PDP는 명칭을 통해서 확인할 수 있듯이, Plot을 그리는 방법이다. 시각화는 가장 강력한 해석 방법 중 하나로 PDP는 $$\mathbf{X}$$ 변수 집합에 대하여 반응변수 $$\mathbf{y}$$의 변화 수준을 표현한다. 그러나 안타깝게도 변수 1~2개 수준에서만 표현이 가능하다.

$$\mathbf{X} = \{x_{1},...,x_{p}\}$$ 와 같이 p차원의 변수가 존재하고 모델 $$\hat{F}(\mathbf{X})$$를 통해 반응변수 $$\mathbf{y}$$를 예측하게 된다. 변수 $$\mathbf{X}$$는 2가지 부분집합으로 나눌 수 있다. 첫번째 집합은 우리가 살펴보고자 하는 변수를 모은 관심변수 집합 $$z_{s}$$ 이고 또 다른 집합은 $$z_{c}$$이다. 이 때, 두 집합은 교집합이 존재하지 않는다. 따라서 $$z_{s} \cup z_{c} = \mathbf{X}$$ 가 성립된다.

만약 $$z_{c}$$ 변수가 한가지 특정한 값으로 고정된 상태라면, $$\hat{F}(\mathbf{X})$$ 함수는 오로지 관심변수 집합 $$z_{s}$$ 만을 변수로 갖는 함수가 된다.

$$ \hat{F}_{z_{c}}(z_{s}) = \hat{F}(z_{s}\vert z_{c}) $$

일반적으로 함수 $$ \hat{F}_{z_{c}}(z_{s}) $$의 형태는 특정한 값으로 지정한 $$z_{c}$$ 값에 의존하지만, 이러한 의존성은 $$z_{c}$$의 평균치로 대체하는 것 대비 크게 강하지 않다.




#### 참조 문헌
1. [위키백과 오픈 API](https://ko.wikipedia.org/wiki/%EC%98%A4%ED%94%88_API) <br>
2. [직장인을 위한 데이터분석 실무 with 파이썬](https://wikibook.co.kr/playwithdata/)
