---
layout: post
title:  "MultiClass Logistic Model"
date: 2021-01-24
author: seolbluewings
categories: 선형모델
---

[작성중...]

앞선 포스팅에서 우리는 y의 Class가 $$y \in \{0,1\}$$ 인 이진 분류(binary case)에서 Logistic Regression을 수행하는 것을 살펴보았다. 그런데 세상에는 Class를 단 2가지로만 분류하는 문제만 존재하는게 아니다. 3가지 이상의 Class를 분류하는 문제 상황에서 우리는 MultiClass Logistic Model을 활용할 수 있다. 범주가 3개 이상인 상황에서 우리는 반응변수 y가 각 클래스에 속할 확률을 바탕으로 y의 클래스를 예측한다. 즉, $$y \in \{1,2,...,K\}$$ 인 상황에서 우리는 각 $$p(y=i\vert X), \; i=1,2,...,K$$ 를 구해서 가장 확률이 높은 Class 값으로 y의 Class를 예측하게 된다.

또한 y가 갖는 범주의 특성을 살펴보고 문제에 접근해야 한다. y가 단순히 명목형 데이터인 경우에서의 (예를 들자면, 혈액형을 예측하는) Logistic Model이 있을 것이고, y가 순서형 데이터일 때의 (예를 들자면, "매우 나쁨","나쁨","보통","좋음","아주 좋음" 등과 같은 범주) Logistic Model이 있을 것이다. 순서형 반응변수는 순서 정보를 무시하고 단순한 명목형 데이터로 취급하여 모형을 수행할 수 있는데 역으로 명목형 취급변수를 순서형 반응변수로 취급하여 모형화하는 것은 불가능 하다.

#### Multinomial Logistic Model

반응변수가 명목형 변수이면서 동시에 반응변수가 가질 수 있는 값이 3가지 이상일 대 활용되는 방법이다. 데이터 $$\mathbf{X}$$를 바탕으로 예측변수 y가 각 Class에 속할 확률을 추정한다.






#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
