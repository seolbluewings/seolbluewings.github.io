---
layout: post
title:  "Model Evaluation Metrics"
date: 2021-01-13
author: seolbluewings
comments : true
categories: 모델평가
---

모델을 생성하는 과정에서 우리는 예측결과가 가장 좋은 모델을 생성하길 희망한다. 어떠한 예측 모델은 1가지 방법론으로만 만들어지는게 아니다. 분류 분석 문제에서 로지스틱 회귀를 사용할 수도 있고 랜덤 포레스트를 활용할 수도 있다. 즉, 하나의 문제를 해결하기 위해서 다양한 방법론을 활용할 수 있다.

게다가 한가지 방법론 내에서도 hyperparameter 값에 따라 생성가능한 모델의 개수는 무수히 많아질 수 있다. 그렇다면, 우리는 수많은 모델들 중에서 최적의 모델을 골라낼 수 있는 기준을 마련해야할 것이다.

기준을 설정하고 각 모델을 기준에 따라 비교하는 것을 모델 성능 평가라고 한다. 모델의 우수성을 판단할 수 있는 기준은 데이터 분석의 목적에 따라 달라진다. 따라서 우리는 다양한 방식의 모델 성능 평가 기준을 알고있어야 한다.

#### MSE

MSE(평균제곱오차)는 가장 대표적인 평가 기준이다. 가장 단순한 형태인 회귀분석을 진행한다고 가정해보자. 예를 들면, 고객 정보를 가지고 이 고객의 월카드사용금액을 예측하는 모델을 만들어낸다고 하자.

$$\mathcal{D}=\{(x_{1},y_{1}),...,(x_{n},y_{n})\}$$ 과 같이 데이터가 존재한다고 하자. 이 때, $$y_{i}$$는 $$x_{i}$$에 매칭되는 결과값이다. $$x_{i}$$는 고객의 정보일 것이고 $$y_{i}$$는 월카드사용금액인 셈이다. 예측을 위한 모델을 $$f$$ 라고 하자. 예측 모델의 성능을 측정하기 위해서는 예측값인 $$f(x) = \hat{y}$$ 값과 정답 $$y$$를 비교해야 한다.

이러한 상황에서 가장 빈번하게 쓰이는 성능 측정기준이 MSE(평균제곱오차)이며 다음과 같이 계산한다.

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y-\hat{y})^{2}$$

MSE 수식을 통해 확인할 수 있듯이, MSE는 정답에 가깝게 예측할 수록 그 값이 작아진다.

#### Accuracy

분류 문제에서는 MSE가 아닌 정확도(Accuracy)나 오차율(Error Rate)을 기준으로 삼는다. 이진분류, 다중분류 모두 활용 가능한 지표이다.

Accuaracy는 전체 데이터 중에서 정확하게 예측한 데이터 개수의 비율을 의미한다. Error Rate는 이와 반대로 전체 데이터 중에서 잘못 예측한 데이터 개수의 비율을 의미한다.

마찬가지로 $$\mathcal{D}=\{(x_{1},y_{1}),...,(x_{n},y_{n})\}$$ 과 같이 데이터가 존재한다고 하자. 대신 이 때 $$y_{i} \in \{0,1\} $$ 이라 하자. Accuracy와 Error Rate는 다음과 같이 계산할 수 있으며, Error Rate는 1-Accuarcy 값임을 알 수 있다.

$$
\begin{align}
\text{Accuracy} &= \frac{1}{n}\sum_{i=1}^{n}\mathbb{I}(\hat{y} = y) \nonumber \\
\text{Error Rate} &= \frac{1}{n}\sum_{i=1}^{n}\mathbb{I}(\hat{y} \neq y) \nonumber
\end{align}
$$

#### Recall, Precision, F1 Score

이 포스팅에서 가장 중요한 부분이다. 오차율과 정확도만으로 충분하지 않을 때가 있다. 다음과 같은 상황을 가정해보자.

채무불이행 인원을 예측하는 모델을 만들어보고자 한다. 대출시행 후, 실제 채무 불이행으로 이어질 확률이 2%라고 한다면 정확도만으로는 모델의 성능을 평가할 수 없다. 모델이 만약 모든 인원에 대해 정상적으로 돈을 갚을 것이라 예측한다면, 98%의 확률로 정답을 맞출 수 있기 때문이다.

이러한 상황에서는, 예측 모델이 실제 채무불이행인 사람을 채무불이행으로 예측할 확률 등이 더욱 중요한 관심사라고 할 수 있다. 그리고 채무불이행으로 예측되는 인원들 중에서 실제로 채무불이행인 사람의 비율 역시 중요한 지표라고 할 수 있겠다. 이러한 상황에서 활용되는 지표가 바로 정밀도(Precision)와 재현율(Recall)이다.

이진분류 문제에서 우리는 아래의 표와 같은 결과를 얻을 수 있다. 이러한 형태의 표를 혼동행렬(Confusion Matrix)라고 부른다.

||예측 Y|예측 N|
|:---:|:---:|:---:|
|실제 Y|TP(True Positive)|FN(False Negative)|
|실제 N|FP(False Positive)|TN(True Negative)|

이 Confusion Matrix를 활용하여 정밀도(Precision)와 재현율(Recall)은 다음과 같이 구할 수 있다.

$$
\begin{align}
\text{Precision} &= \frac{\text{TP}}{\text{TP}+\text{FP}} \nonumber \\
\text{Recall} &= \frac{\text{TP}}{\text{TP}+\text{FN}} \nonumber
\end{align}
$$

Precision과 Recall은 서로 Trade-Off 관계에 있다. 앞서 언급했던 채무불이행 예측모델을 활용해 논의를 이어가자. 모델을 통해 실제 채무불이행인 사람을 최대한 많이 골라내려면(Recall을 높이려고 한다면), 모델이 최대한 많은 사람을 채무불이행으로 판단하도록 만들면 된다. 이러한 경우에는 Precision이 떨어진다.

반대로 채무불이행 예측인원 중 실제 채무불이행인 사람의 비율을 높이고자 한다면(Precision을 높이려고 한다면), 특정인원을 채무불이행으로 판단하는데 신중해져야 한다. 이러한 경우에는 Recall값이 떨어진다.

Precision과 Recall의 중요성은 문제의 상황마다 다르다. 상품추천 시스템에서는 사용자가 싫어할 내용을 최대한 배제시키면서 사용자가 흥미를 느낄 컨텐츠를 추천해야 한다. 이 경우는 Precision이 중요하다. 반면, 채무불이행 예측의 경우에는 최대한 채무불이행 인원을 놓치지 않는 것이 중요하다. 이 경우는 Recall이 중요한 것이다.

Precision과 Recall의 조화평균인 F1 Score도 있다. F1 Score에 대한 일반식은 다음과 같다. Precision을 P, Recall을 R로 표기하기로 한다.

$$
F_{\beta} = \frac{(1+\beta^{2})PR}{(\beta^{2}P)+R}, \quad \text{where} \beta >0
$$

여기서 $$\beta$$는 Precision에 대한 Recall의 상대적 중요도를 의미한다. 만약 $$\beta>1$$인 경우, Recall의 영향력이 크고 $$\beta<1$$인 경우는 Precision의 영향력이 크다. $$\beta=1$$이라면, 이를 F1 Score라 부르며 이는 Precision과 Recall의 조화평균이다.

#### ROC와 AUC





#### 참조 문헌

1. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)