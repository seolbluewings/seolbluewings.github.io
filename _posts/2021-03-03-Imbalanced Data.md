---
layout: post
title:  "Overcoming Imbalanced Class Data"
date: 2021-03-03
author: seolbluewings
categories: 모델평가
---

분류 문제를 해결하려는 과정에서 우리는 target data인 $$\mathbf{y}$$의 Class 불균형 이슈를 빈번하게 접하게 된다. 앞선 [모델평가 방법론](https://seolbluewings.github.io/%EB%AA%A8%EB%8D%B8%ED%8F%89%EA%B0%80/2021/01/13/Model-Evaluation-Metrics.html) 포스팅에서 소개한 바와 같이 데이터 Class 불균형은 분류 목적 알고리즘 성능에 영향을 미치게 된다.

이러한 상황을 가정해 볼 수 있다. 2020년 카드사의 평균 연체율은 1.4% 수준이다. 따라서 이러한 상황에서는 모든 고객을 대상으로 연체가 발생하지 않을 것이라 예상해도 무려 98.6%의 정확도(Accuracy)를 확보할 수 있다. 가장 중요한 것은 실제 연체가 발생한 회원을 연체할 것으로 예측할 재현율(Recall)이 중요하다. 이처럼 불균형 데이터는 우리가 결과를 해석하는데 애를 먹게 만든다.

따라서 불균형 데이터를 활용하기 이전에 불균형 문제를 해결하고 모델링 작업에 들어가는 것도 이러한 이슈를 해결할 수 있는 하나의 방법이 될 수 있다. 대표적인 방법으로 UnderSampling, OverSampling, SMOTE, Borderline-SMOTE 정도가 있는데 앞선 2개는 간단하게 살펴보고 뒤의 2개를 이번 포스팅에서 중점적으로 다뤄보고자 한다.

#### UnderSampling과 OverSampling

![ID](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Sampling.png?raw=true){: .align-center}{:width="80%" height="80%"}

과소표집(UnderSampling)과 과대표집(OverSampling)은 위의 그림을 통해 이해할 수 있다.

먼저 UnderSampling은 두가지 Class 중에서 다수의 Class(Majority Class, 이하 $$S_{maj}$$) 데이터를 줄이는 방법이다. 많은 쪽의 데이터를 줄여서 Class의 균형을 맞출 수 있지만, 정보가 버려진다는 것의 단점이 있다.

OverSampling은 소수의 Class(Minority Class, 이하 $$S_{min}$$)의 데이터를 복제하여 Class 간의 균형을 맞춘다. 그러나 이러한 방법은 같은 데이터 포인트를 중복해서 활용하는 방법이기 때문에 Overfitting의 이슈가 있는 방법이다.

두가지 방법 모두 Class 불균형을 해소할 수 있는 방법이나 그리 좋아보이지는 않는다. 특히 UnderSampling의 경우는 일부 데이터를 버리기 때문에 사용이 꺼려지는 부분이 있다. 따라서 OverSampling을 활용하되 성능을 개선시킨 것들이 이후 소개하고자 하는 SMOTE 와 Borderline-SMOTE가 되겠다.

#### SMOTE(Synthetic Minority Over-Sampling Technique)

SMOTE 알고리즘은 데이터 수가 부족한 $$S_{min}$$의 데이터 개수를 증가시켜 각 Class의 데이터 개수를 적절하게 유지시키는 방법으로 OverSampling과 비슷한 전략을 취한다고 볼 수 있다.

SMOTE와 OverSampling의 가장 큰 차이는 다음과 같다. 기존의 OverSampling은 $$S_{min}$$ 데이터를 동질하게 복제함으로써 $$S_{min}$$에 속하는 데이터 개수를 증가시키지만, SMOTE는 기존의 $$S_{min}$$ 데이터를 활용하여 새로운 데이터를 생성하고 신규 생성된 데이터를 $$S_{min}$$ 데이터로 간주하고 이를 Train Data Set에 포함시킨다.

SMOTE 알고리즘은 다음의 과정을 통해 새로운 데이터를 생성해낸다.

1. 원 데이터에서 $$S_{min}$$에 속하는 데이터 개수가 N이라고 한다면, N개의 데이터의 Subset인 M개 데이터를 선택한다. 즉, $$N \geq M$$
2. 아래의 과정을 $$ i = 1,2,...,M $$ 에 대해 반복한다.
3. 하나의 데이터 포인트 $$x_{i}$$에 대한 K-NN(nearest-neighbors)을 찾는다.
4. $$x_{i}$$의 K-NN이 되는 포인트들 중 1개 $$x_{j}$$를 임의(random)로 선택한다.
5. 두 데이터 포인트 간의 유클리드 거리, $$\vert\vert x_{i} - x_{j} \vert\vert$$ 를 구한다.
6. 0~1 사이의 숫자중 하나를 임의로 선택한다. 즉, $$\text{Uniform}(0,1)$$에서 임의의 값 u를 하나 추출한다.
7. 앞서 구한 유클리드 거리에 6번째 단계에서 추출한 값을 곱하여, $$ u\times\vert\vert x_{i} - x_{j} \vert\vert$$, 이를 $$x_{i}$$에 더한다.
8. 7번째 단계에서 구한 값을 새로운 $$S_{min}$$ 데이터로 간주하고 이를 모델 학습 시, Train Data Set에 포함시킨다.

![ID](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/SMOTE.png?raw=true){: .align-center}{:width="70%" height="70%"}

1~8 단계를 거치면 그림과 같이 가상의 $$S_{min}$$ 데이터를 만들어낼 수 있다. 그림은 $$K=5$$ 일 때의 케이스이다.

#### Borderline-SMOTE

앞서 소개한 SMOTE는 새로운 데이터를 합성하는 과정에서 $$S_{min}$$ 데이터만 고려할 뿐, $$S_{maj}$$ 데이터는 완전히 배제한다. 기존의 SMOTE 알고리즘이 $$S_{min}$$에 포함되는 데이터 중 하나를 임의로 하나 선택한다면, 새로운 Borderline-SMOTE 알고리즘은 $$S_{min}$$에 속하는 데이터 포인트들 중 모델을 학습하는데 유의할 것으로 판단되는 것들을 골라서 새로운 데이터를 합성해낸다.

![ID](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/SMOTE1.png?raw=true){: .align-center}{:width="70%" height="70%"}

그림과 같이 Borderline-SMOTE는 $$S_{min}$$과 $$S_{maj}$$ 경계에 있는 $$S_{min}$$ 데이터들만 대상으로 데이터를 합성하여 분류하기 어려운 부분을 더 잘 학습하기 위한 가상의 데이터를 만들어낸다. Borderline-SMOTE는 다음과 같이 새로운 합성 데이터를 생성해낸다.

1. 원 데이터에서 $$S_{min}$$에 속하는 데이터 개수가 N이라고 한다면, N개의 데이터의 Subset인 M개 데이터를 선택한다. 즉, $$N \geq M$$
2. 아래의 과정을 $$ i = 1,2,...,M $$ 에 대해 반복한다.
3. 하나의 데이터 포인트 $$x_{i}$$에 대한 K-NN을 구한다. 이 때, $$S_{min}$$과 $$S_{maj}$$ 를 모두 포함하여 K-NN을 구한다.
4. $$x_{i}$$의 K-NN 중에서 $$S_{maj}$$에 속하는 데이터 개수($$\alpha$$)를 확인한다.
5. 여러 $$x_{i}$$들 중에서 $$\frac{K}{2}\leq \alpha \leq K$$를 만족하는 데이터 포인트만 선정한다.
6. 이후의 과정은 SMOTE와 동일하게 새로운 합성 데이터를 생성하며 이 때는 $$S_{min}$$만 고려한다.

만약 어떠한 한 점 $$x_{i}$$의 K-NN이 모두 $$S_{maj}$$에 속한다면, 그 점은 Noise로 간주하며 5번의 조건을 만족시키지 못한다면 그 데이터는 데이터를 추가 생성하지 않더라도 이미 잘 분류해낼 수 있는 공간에 데이터가 존재하고 있다고 간주할 수 있다.

즉, Borderline-SMOTE는 기존의 SMOTE와 비교하여 분류가 까다로운 부분을 더 잘 맞출 수 있도록 하기위한 SMOTE의 한 단계 업그레이드 된 버전이라고 볼 수 있다.



#### 참조 문헌
1. [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) <br>
2. [Borderline-SMOTE](https://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf)
