---
layout: post
title:  "Naive Bayes Classifier"
date: 2020-12-28
author: seolbluewings
comments : true
categories: Bayesian
---

[작성중...]

나이브 베이즈 분류기(Naive Bayes Classifier)는 베이즈 정리에 기반하여, 즉 조건부 확률을 이용해 각 Class에 속할 확률을 계산하여 분류를 진행하는 학습 모델을 의미한다.

Naive Bayes Classifier는 Naive란 단어가 포함되는 명칭에서 유추할 수 있듯이 가장 단순한 형태의 가정에서 출발한 모델이다. Naive Bayes Classifier는 데이터셋의 모든 컬럼이 동등한 조건으로 영향력을 행사하고 서로 독립적이라는 가정을 바탕으로 시작하는 모델이다.

그런데 사실 모델을 만드는 과정에서 모든 변수의 중요성이 동등하고 상호 독립적인 경우는 거의 없다. 우리가 채무 불이행을 예측하는 모델을 만든다고 생각해보자. 상식적으로 개개인의 소득수준이 성별보다 더 중요한 변수일 것이고 각자의 소득수준은 개인의 학력과 연관이 있을 가능성이 크다. 아닌 사람들도 있겠지만, 대다수 사람들이 생각하기에는 지금 이야기한 관계들이 성립한다고 자연스럽게 받아들인다.

앞서 말했듯이 Naive Bayes Classifier는 이러한 관계를 다 무시한다. 즉, 가장 단순한 가정으로 모델을 생성하기 때문에 필자는 강의시간에 분류 모델을 만들고자 할 때, 무조건 Naive Bayes Classifier보다는 좋은 모델을 만들어야 한다는 이야기를 들은 적이 있다. 상식적으로 받아들일 수 있는 분류기 효율성의 하한(lower bound)을 구하는 모델로 학습과정에서 이해하였고 그런 의미에서 Naive Bayes Classifier가 어떤 원리로 만들어지는지 알 필요가 있다고 생각 된다.

#### 베이즈 정리(Bayes Rule)

#### Naive Bayes Classifier 


#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)