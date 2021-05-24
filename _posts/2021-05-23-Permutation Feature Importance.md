---
layout: post
title:  "Permutation Feature Importance"
date: 2021-05-23
author: seolbluewings
categories: Statistics
---

[작성중...]

[PDP](https://seolbluewings.github.io/statistics/2021/05/09/Partial-Dependence-Plot.html)처럼 변수 중요도를 파악할 수 있는 방법은 여러가지 있다. 이번 포스팅에서는 PDP처럼 변수의 중요도 파악에 자주 활용되는 방법인 Permutation Feature Importance(이하 PFI)를 소개하고자 한다.

변수의 중요도를 파악할 때, 우리는 변수의 존재 유무에 따른 모델의 성능을 비교하는 가장 직관적인 방법을 선택할 수 있다.

변수 $$\mathbf{X} = \{x_{1},...,x_{n}\}$$ 에 대해서 $$x_{i}$$의 중요도를 파악하길 희망한다고 가정하자. 앞으로 우리가 중요도를 파악하고자 선정하는 변수는 모두 $$x_{i}$$로 표기한다.

먼저 $$\mathbf{X}$$ 변수를 통으로 집어넣은 모델 A을 생성하고 A 모델의 성능을 평가한 다음 $$x_{i}$$를 제외한 나머지 변수들을 모두 집어넣은 모델 B를 생성하여 B의 성능을 A와 비교하는 방법이 있다.

그러나 PFI 방식은 하나의 변수 $$x_{i}$$ 를 완전히 제거하여 모델을 생성하는 방법 대신, $$x_{i}$$ 변수의 데이터 순서를 무작위로 섞어(permutation) 해당 변수를 기존 변수에서 noise나 다름없게 만드는 작업을 가한다.

훈련 데이터를 통해 모델을 학습한 뒤, 검증 데이터에서 변수 $$x_{i}$$ 를 무작위로 데이터를 섞어버린 뒤 학습한 모델에 fitting하여 모델의 성능을 파악해본다.

만약 $$x_{i}$$ 변수의 순서를 무작위로 섞었을 때, 모델의 성능이 떨어진다면 이 $$x_{i}$$ 변수는 중요 변수라고 판단을 하게 된다. 변수 $$x_{i}$$가 의미없는 noise 변수로 작동한 결과 모델의 성능이 떨어진 것으로 직관적인 판단이 가능하다.

그러나 변수 $$x_{i}$$의 순서를 무작위로 섞었음에도 모델의 성능이 떨어지지 않는다면, 변수 $$x_{i}$$의 중요성은 크지 않다고 해석이 가능하다. 총 n개의 변수가 존재하는 상황에서는 n개의 변수에 대해 Permutation을 시행한 후, 그 중요도를 순서대로 정렬할 수 있다.






PDP에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Partial%20Dependence%20Plot.ipynb)에서 확인 가능하다.

#### 참조 문헌
1. [Partial Dependence Plot (PDP)](https://christophm.github.io/interpretable-ml-book/pdp.html) <br>
2. [ "Greedy function approximation: A gradient boosting machine." Annals of statistics (2001)](http://scholar.google.co.kr/scholar_url?url=https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451&hl=ko&sa=X&ei=gL-gYOqqMIqWyQTCj5CoDQ&scisig=AAGBfm32i0MEcGQztHTLEV3WO3VYfi3h9g&nossl=1&oi=scholarr)
