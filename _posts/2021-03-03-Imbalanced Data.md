---
layout: post
title:  "Overcoming Imbalanced Class Data"
date: 2021-03-03
author: seolbluewings
categories: 모델평가
---

[작성중...]

분류 문제를 해결하려는 과정에서 우리는 target data인 $$\mathbf{y}$$의 Class 불균형 이슈를 빈번하게 접하게 된다. 앞선 [모델평가 방법론](https://seolbluewings.github.io/%EB%AA%A8%EB%8D%B8%ED%8F%89%EA%B0%80/2021/01/13/Model-Evaluation-Metrics.html) 포스팅에서 소개한 바와 같이 데이터 Class 불균형은 분류 목적 알고리즘 성능에 영향을 미치게 된다.

이러한 상황을 가정해 볼 수 있다. 2020년 카드사의 평균 연체율은 1.4% 수준이다. 따라서 이러한 상황에서는 모든 고객을 대상으로 연체가 발생하지 않을 것이라 예상해도 무려 98.6%의 정확도(Accuracy)를 확보할 수 있다. 가장 중요한 것은 실제 연체가 발생한 회원을 연체할 것으로 예측할 재현율(Recall)이 중요하다. 이처럼 불균형 데이터는 우리가 결과를 해석하는데 애를 먹게 만든다.

따라서 불균형 데이터를 활용하기 이전에 불균형 문제를 해결하고 모델링 작업에 들어가는 것도 이러한 이슈를 해결할 수 있는 하나의 방법이 될 수 있다. 대표적인 방법으로 UnderSampling, OverSampling, SMOTE, ADASYN 정도가 있는데 앞선 2개는 간단하게 살펴보고 뒤의 2개를 이번 포스팅에서 중점적으로 다뤄보고자 한다.

#### UnderSampling과 OverSampling

![ID](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Sampling.png?raw=true){: .align-center}{:width="70%" height="70%"}

과소표집(UnderSampling)과 과대표집(OverSampling)은 위의 그림을 통해 이해할 수 있다.

먼저 UnderSampling은 두가지 Class 중에서 다수의 Class(Majority Class) 데이터를 줄이는 방법이다. 많은 쪽의 데이터를 줄여서 Class의 균형을 맞출 수 있지만, 정보가 버려진다는 것의 단점이 있다.

OverSampling은 소수의 Class(Minority Class)의 데이터를 복제하여 Class 간의 균형을 맞춘다. 그러나 이러한 방법은 같은 데이터 포인트를 중복해서 활용하는 방법이기 때문에 Overfitting의 이슈가 있는 방법이다.

두가지 방법 모두 Class 불균형을 해소할 수 있는 방법이나 그리 좋아보이지는 않는다. 특히 UnderSampling의 경우는 일부 데이터를 버리기 때문에 사용이 꺼려지는 부분이 있다. 따라서 OverSampling을 활용하되 성능을 개선시킨 것들이 이후 소개하고자 하는 SMOTE 와 ADASYN이 되겠다.

#### SMOTE(Synthetic Minority Over-Sampling Technique)

SMOTE 알고리즘은 데이터 수가 부족한 Minority Class의 데이터 개수를 증가시켜 각 Class의 데이터 개수를 적절하게 유지시키는 방법으로 OverSampling과 비슷한 전략을 취한다고 볼 수 있다.

OverSampling은 Minority Class 데이터를 중복해서 활용하지만, SMOTE는 Minority Class 데이터를 활용하여 Minory Class로 간주할 가상의 새로운 데이터 포인트를 만들어내는 방법이다. 따라서 SMOTE를 통해 Minority Class의 데이터 개수를 늘리고 이를 통해 각 Class 별 데이터 수를 적절하게 맞춰줌으로써 데이터 불균형 이슈를 해결할 수 있다.

(추후 작성...)



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
