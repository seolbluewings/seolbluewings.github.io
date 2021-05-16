---
layout: post
title:  "Partial Dependence Plot"
date: 2021-05-09
author: seolbluewings
categories: Statistics
---

모델은 보통 2가지 관점에서 평가를 받는다. 관점 하나는 설명력이고 또 다른 관점 하나는 예측력이다. 보통 예측력이 좋은 모델일수록 해석력이 떨어지는 것으로 알려져 있다. 여기서 모델을 해석한다는 것은 변수 $$\mathbf{X}$$가 1단위 증가할 때, 반응변수 $$\mathbf{y}$$가 얼마나 변하는가? 를 알아내는 것을 의미한다. 결국 이는 Input에 의한 Output의 변동성을 체크한다는 것이다.

그러나 앙상블 모형에서부터 딥러닝까지 모델이 복잡해질수록 예측력은 높아지나 Input의 변동이 Output 변동에 얼마나 영향을 미쳤는지를 파악하기가 점점 어려워진다. 이렇게 예측력이 높으나 해석력이 떨어지는 모델을 Black Box 모델이라 부른다.

회귀분석의 경우, 회귀계수 $$\beta$$를 통해 Input 변화에 대한 Output 변화의 수준, 방향성을 파악할 수 있다. 이 방식은 Input $$\mathbf{X}$$가 1단위 증가할 때, $$\mathbf{y}$$가 얼마나 변하였는가를 파악하는 것이며 굉장히 직관적인 해석이 가능하다. 또한 우리는 어떤 Input 변수가 Output에 가장 영향을 많이 주었는지 파악할 수 있고 결과적으로 Input 변수의 상대적 중요도(Feature Importance)를 파악할 수 있다.

Black Box 모델에서도 Input 변수의 변동에 따른 Output의 변화를 살펴보고 싶은 욕구가 생기기 마련인데 이러한 이슈를 해결하기 위해 제안된 방법이 바로 Partial Dependence Plot(PDP) 이다.

#### Concept of PDP(Partial Dependence Plot)

PDP는 명칭을 통해서 확인할 수 있듯이, Plot을 그리는 방법이다. 시각화는 가장 강력한 해석 방법 중 하나로 PDP는 $$\mathbf{X}$$ 변수 집합에 대하여 반응변수 $$\mathbf{y}$$의 변화 수준을 표현한다. 그러나 안타깝게도 변수 1~2개 수준에서만 표현이 가능하다.

$$\mathbf{X} = \{x_{1},...,x_{p}\}$$ 와 같이 p차원의 변수가 존재하고 모델 $$\hat{F}(\mathbf{X})$$를 통해 반응변수 $$\mathbf{y}$$를 예측하게 된다. 변수 $$\mathbf{X}$$는 2가지 부분집합으로 나눌 수 있다. 첫번째 집합은 우리가 살펴보고자 하는 변수를 모은 관심변수 집합 $$z_{s}$$ 이고 또 다른 집합은 $$z_{c}$$이다. 이 때, 두 집합은 교집합이 존재하지 않는다. 따라서 $$z_{s} \cup z_{c} = \mathbf{X}$$ 가 성립된다.

만약 $$z_{c}$$ 변수가 한가지 특정한 값으로 고정된 상태라면, $$\hat{F}(\mathbf{X})$$ 함수는 오로지 관심변수 집합 $$z_{s}$$ 만을 변수로 갖는 함수가 된다.

$$ \hat{F}_{z_{c}}(z_{s}) = \hat{F}(z_{s}\vert z_{c}=c) $$

일반적으로 함수 $$ \hat{F}_{z_{c}}(z_{s}) $$의 형태는 특정한 값으로 지정한 $$z_{c}$$ 값에 의존하지만, 이러한 의존성은 $$z_{c}$$의 평균치로 대체하는 것 대비 크게 강하지 않다. 따라서 관심변수 $$z_{s}$$에 대한 모델 $$\hat{F}$$의 Partial Dependence는 다음과 같이 표현이 가능하다.

$$
\begin{align}
p(z_{s}) &= \mathbb{E}_{z_{c}}\left[\hat{F}(z_{s},z_{c})\right] \nonumber \\
&= \int\hat{F}(z_{s},z_{c})p_{z_{c}}(z_{c})dz_{c} \nonumber
\end{align}
$$

여기서 $$ p_{z_{c}}(z_{c}) $$ 란 $$z_{c}$$에 대한 marginal probability density로서 다음의 식을 통해 구할 수 있으며 $$p(z_{s},z_{c})$$는 모든 Input 변수 X에 대한 Joint Distribution이다.

$$ p_{z_{c}}(z_{c}) = \int p(z_{s},z_{c})dz_{s} $$

그러나 일반적으로 $$ p_{z_{c}}(z_{c}) $$ 에 대한 명확한 수식을 알고있는 경우는 극히 드물다. 따라서 모델을 학습하는 과정에서 주어진 데이터를 바탕으로 근사적인 방법을 통해 $$p(z_{s})$$ 를 추정하게 된다.

모델 학습과정에서 사용하는 데이터가 n개 존재한다고 가정하자. 그렇다면 $$z_{c}$$ 집합에 포함되는 변수들도 각 n개씩 존재하여 다음과 같이 표현이 가능할 것이다.

$$ z_{c} = \{z_{1c},z_{2c},...,z_{nc}\}$$

결국 관심변수 $$z_{s}$$에 대한 Partial Dependence를 구하는 것은 $$p(z_{s})$$를 구하는 것이며 이는 모델 학습과정에서 사용되는 $$z_{c} = \{z_{1c},z_{2c},...,z_{nc}\}$$ 데이터를 모두 활용하여 모델 $$\hat{F}$$ 에 적용시킨 후 모델 결과값에 대한 평균을 취해 구하게 된다.

$$ p(z_{s}) = \frac{1}{n}\sum_{i=1}^{n}\hat{F}(z_{s},z_{ic}) $$

이 상황에서 관심변수 $$z_{s}$$ 의 크기를 조정함에 따라 예측 결과(모델 적합결과) $$\hat{F}$$를 파악하는 것이 변수 $$z_{s}$$ 에 대한 Partial Dependence를 구하는 과정이며 이 변화를 Plot으로 표현한 것이 Partial Dependence Plot 이다.

#### PDP의 장/단점

PDP는 무엇보다 이미지로 결과를 보여준다는 점에서 해석이 직관적이다. 또한 인과적인 해석이 가능하다. PDP를 통해 표현하려하는 관심변수가 다른 변수들과 상관성이 낮다면 PDP는 관심변수가 결과 예측에 미치는 평균적인 영향력을 잘 보여준다.

다만, 시각화를 통해 보여준다는 점으로 인해 단점이 있다. 사람이 시각적으로 3차원까지 밖에 인지할 수 없기 때문에 PDP는 최대 2개 변수까지만 관심변수를 설정할 수 있다.

또한 변수의 독립성 측면에서도 이슈가 있다. Partial Dependence는 계산하고자 하는 변수가 다른 변수와의 Correlation이 없다는 상황을 가정한다. 다음과 같은 상황을 생각해보자. 사람의 달리기 속도$$(\mathbf{y})$$를 예측한다고 하는 상황에 $$\mathbf{X}$$ 변수로 키와 몸무게가 주어졌다고 하자. 이 때 우리는 달리기 속도에 대해 몸무게의 영향력을 파악하기 위해 Partial Dependence를 계산하려 하는데 상식적으로 몸무게와 키는 상관성이 있는 변수이다.

몸무게 40kg인 경우에 대해 Partial Dependence를 구하려고 하는 상황에서 우리는 키값들의 평균을 구하게 되는데 평균을 구하는 과정에서 키가 190cm인 사람이 들어갈 수도 있다. 몸무게 40kg인 사람에게는 비현실적인 수치가 계산에 포함되는 것이다. 이러한 문제를 해결하기 위해 [Accumulated Local Effect Plots](https://christophm.github.io/interpretable-ml-book/ale.html) 이란 방법을 사용하기도 한다.



#### 참조 문헌
1. [Partial Dependence Plot (PDP)](https://christophm.github.io/interpretable-ml-book/pdp.html) <br>
2. [ "Greedy function approximation: A gradient boosting machine." Annals of statistics (2001)](http://scholar.google.co.kr/scholar_url?url=https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451&hl=ko&sa=X&ei=gL-gYOqqMIqWyQTCj5CoDQ&scisig=AAGBfm32i0MEcGQztHTLEV3WO3VYfi3h9g&nossl=1&oi=scholarr)
