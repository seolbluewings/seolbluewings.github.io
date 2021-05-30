---
layout: post
title:  "Permutation Feature Importance"
date: 2021-05-23
author: seolbluewings
categories: Statistics
---

[PDP](https://seolbluewings.github.io/statistics/2021/05/09/Partial-Dependence-Plot.html)처럼 변수 중요도를 파악할 수 있는 방법은 여러가지 있다. 이번 포스팅에서는 PDP처럼 변수의 중요도 파악에 자주 활용되는 방법인 Permutation Feature Importance(이하 PFI)를 소개하고자 한다.

변수의 중요도를 파악할 때, 우리는 변수의 존재 유무에 따른 모델의 성능을 비교하는 가장 직관적인 방법을 선택할 수 있다.

변수 $$\mathbf{X} = \{x_{1},...,x_{n}\}$$ 에 대해서 $$x_{i}$$의 중요도를 파악하길 희망한다고 가정하자. 앞으로 우리가 중요도를 파악하고자 선정하는 변수는 모두 $$x_{i}$$로 표기한다.

먼저 $$\mathbf{X}$$ 변수를 통으로 집어넣은 모델 A을 생성하고 A 모델의 성능을 평가한 다음 $$x_{i}$$를 제외한 나머지 변수들을 모두 집어넣은 모델 B를 생성하여 B의 성능을 A와 비교하는 방법이 있다.

그러나 PFI 방식은 하나의 변수 $$x_{i}$$ 를 완전히 제거하여 모델을 생성하는 방법 대신, $$x_{i}$$ 변수의 데이터 순서를 무작위로 섞어(permutation) 해당 변수를 기존 변수에서 noise나 다름없게 만드는 작업을 가한다.

훈련 데이터를 통해 모델을 학습한 뒤, 검증 데이터에서 변수 $$x_{i}$$ 를 무작위로 데이터를 섞어버린 뒤 학습한 모델에 fitting하여 모델의 성능을 파악해본다.


![PFI](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/PFI.png?raw=true){:width="70%" height="70%"}{: .aligncenter}

만약 $$x_{i}$$ 변수의 순서를 무작위로 섞었을 때, 모델의 성능이 떨어진다면 이 $$x_{i}$$ 변수는 중요 변수라고 판단을 하게 된다. 변수 $$x_{i}$$가 의미없는 noise 변수로 작동한 결과 모델의 성능이 떨어진 것으로 직관적인 판단이 가능하다.

그러나 변수 $$x_{i}$$의 순서를 무작위로 섞었음에도 모델의 성능이 떨어지지 않는다면, 변수 $$x_{i}$$의 중요성은 크지 않다고 해석이 가능하다. 총 n개의 변수가 존재하는 상황에서는 n개의 변수에 대해 Permutation을 시행한 후, 그 중요도를 순서대로 정렬할 수 있다.

이 과정을 수식으로 표현하면 다음과 같을 것이다.

훈련 데이터를 통해 학습된 모델을 $$F$$라 표현하고 input 변수를 $$\mathbf{X}= \{x_{1},x_{2},...,x_{p}\}$$ 로 표현하기로 하자 타겟변수를 $$\mathbf{y}$$로 표현하기로 한다.

- 학습된 모델에 대한 최초 error를 계산한다.
$$ e^{ori} = L(\mathbf{y},F(\mathbf{X}))$$
- 각 변수 $$x_{1},...,x_{p}$$에 대하여 다음의 과정을 반복 수행한다.
- $$i$$번째 변수 $$x_{i}$$에 대하여 변수의 순서를 무작위로 바꾼 $$\mathbf{X}^{per}$$를 생성한다. 이렇게 생성된 $$\mathbf{X}^{per}$$는 변수 $$x_{i}$$와 $$\mathbf{y}$$의 관계를 끊어버린다.
- Permutation 된 데이터 $$\mathbf{X}^{per}$$에 대한 error를 계산한다.
$$ e^{per} = L(\mathbf{y},F(\mathbf{X}^{per}))$$
- $$e^{per} / e^{ori}$$ 또는 $$ e^{per} - e^{ori} $$ 를 계산하고 p가지 변수에 대해 내림차순으로 정렬하여 값이 큰 순서대로 변수의 중요도를 부여한다.



#### 랜덤 포레스트의 MDI는?

[랜덤 포레스트](https://seolbluewings.github.io/statistics/2020/03/30/Bagging.html)활용 시 사용하는 대표적인 변수 중요도 계산법인 MDI(Mean Decrease in Impurity) 방식이 있다. MDI는 tree기반 모형에서 각 노드가 분기(split)될 때, 사용되는 변수가 평균적으로 불순도 감소에 어느 수준의 영향력을 미치는가를 고려한다. 불순도 감소 정도가 클수록 그 변수는 중요한 변수로 간주된다.

$$
I_{m}(m) - \frac{N_{c_{1}}}{N_{m}}I_{m}(C_{1})-\frac{N_{c_{2}}}{N_{m}}I_{m}(C_{2})
$$

이 수식에서 $$I_{m}$$은 불순도를 계산하는 함수이며, $$N_{x}$$는 x노드의 관측치 개수를 의미한다. $$N_{m}$$는 부모노드를 의미하며, $$N_{c_{1}},N_{c_{2}}$$ 는 자식 노드를 의미한다.

랜덤 포레스트가 tree를 여러가지 생성하여 이를 평균내는 앙상블 모델이기 때문에 MDI를 통해 생성되는 변수 중요도 Plot은 어느 정도 일반화된 값이라고 볼 수 있다.

MDI는 변수 중요도를 확인할 수 있는 가장 빠른 방법이지만, 이 결과가 정확하다고 확언할 수는 없다. 랜덤 포레스트 모델이 생성될 때마다 MDI를 통해 생성하는 Plot은 매번 결과가 다르며, 랜덤 포레스트가 연속형 변수 or 카테고리 종류가 많은(high-cardinality) 변수에서 변수의 중요도를 과대 평가한다는 것으로 알려져 있다.

#### PFI의 특징 & 장/단점

PFI는 모델 종류를 가리지 않는다. 중요도를 파악하고자하는 변수의 순서를 무작위로 섞는 것은 어느 모델에서든 가능한 행위이다.

변수 별 Partial Importance를 확인하는 PDP와 달리 PFI는 다른 변수와의 교호작용까지 반영한 중요도를 표현한다. 특정 변수의 순서를 랜덤하게 섞으면 상식적으로 다른 변수와의 연결성이 끊어진다고 볼 수 있다. 따라서 순서가 섞이는 변수와 관련있는 모든 교호작용이 사라지게 된다.

PFI는 데이터를 무작위로 섞는 특징으로 인해 MDI와 마찬가지로 실행할 때마다 결과가 달라지는 것은 필연적이다. 따라서 데이터를 섞는 Permutation 횟수를 증가시켜 매 시행마다 결과가 달라지는 것에 대한 분산을 줄이는 방향으로 변동성을 줄여간다. 그러나 이 방법은 연산량 증가로 이어지기 때문에 적절한 Permutation 횟수를 결정지어야 한다.

또한 데이터를 무작위로 섞는 것에 주의할 필요가 있다. 변수간 상관 관계가 존재하는 경우(키-몸무게), 무작위로 데이터를 섞다보면 키 190cm에 몸무게 45kg라는 비현실적인 데이터가 생성될 가능성이 있다. 따라서 PFI 결과 해석 이전에 각 변수가 서로 독립적인지를 따져보아야하며, 변수간 상관 관계가 있다면 이를 고려하면서 결과를 해석할 수 있어야 한다.


PFI에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Partial%20Dependence%20Plot.ipynb)에서 확인 가능하다.

#### 참조 문헌
1. [Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance) <br>
2. [Permutation Feature Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
