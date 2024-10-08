---
layout: post
title:  "Cox Proportional-Hazard Model"
date: 2022-05-01
author: seolbluewings
categories: Statistics
---

고객 이탈 예측을 위해 생존분석의 개념을, 구체적인 모델링을 위해서 Cox Proportional Hazard Model에 대해 알아보고자 한다. Kaplan-Meier 분석은 시간에 따른 특정 집단의 잔존률 분석에 활용되었는데 Cox PH Model은 잔존률에 영향을 미치는 변수를 분석하는 것으로 보유카드 개수가 1개 줄었을 때, 고객이 이탈할 확률이 얼마나 변동되는가? 라는 질문에 대한 답을 찾을 수 있다.

Cox PH 모델은 생존함수와 데이터를 결합한 모델로 어떠한 시점 t까지 잔존한 고객이 직후에 이탈할 확률을 다음과 같이 표현한다.

$$ h(t\vert X = x) = h_{0}(t)\text{exp}(x^{T}\beta) $$

여기서 $$\beta$$는 각 변수에 대한 계수(coefficient)이며, $$h_{0}(t)$$는 $$x=0$$일 때의 baseline hazard function이다. 이 수식에서 우리는 시간 $$t$$가 baseline hazard function에만 존재하는 것에 주목할 필요가 있다.

이는 Cox PH Model의 대표적인 가정인 비례위험 가정과 밀접한 관계가 있다. 비례위험 가정이란 변수가 잔존/이탈에 영향을 주지만, 그 영향은 시간과는 무관하다는 가정이다.

서로 다른 두 대상의 harzard ratio는 시간에 따라 변하는 baseline hazard function과는 독립적이다.

$$ \frac{h(t\vert x_{1})}{h(t\vert x_{2})} = \frac{h_{0}(t)\text{exp}(x_{1}^{T}\beta)}{h_{0}(t)\text{exp}(x_{2}^{T}\beta)} = \text{exp}\left((x_{1}-x_{2})^{T}\beta \right)$$

따라서 모델 학습 과정에서는 baseline hazard function은 무시하고 partial likelihood function을 이용하여 $$\beta$$에 대한 학습이 이루어진다.

$$\beta$$에 대한 추정이 이루어지면, 이를 바탕으로 prediction 수행이 가능해진다. 기존 linear regression 에서 MSE와 같은 평가 지표가 사용된 것처럼 Survival Analysis에서도 Survival Data에 맞는 평가 지표 설정이 필요하다.

Survival Analysis에서 가장 대표적인 지표는 Concordance Index(C-Index)로 대상에 대한 정확한 잔존시간을 예측하는 것보다 여러 대상의 잔존 시간을 상대적으로 비교한다. 추정하는 여러 대상의 이탈 순서를 잘 예측하는지에 보다 초점을 둔 지표라고 할 수 있다.

대상 i에 대해서 $$y_{i}$$가 실제 이탈이 발생한 시간이고 $$\hat{y_{i}}$$가 모델이 예측한 시간이라 한다면, Concordance probability는 $$p\left(\hat{y_{j}}>\hat{y_{i}}\vert y_{j}>y_{i} \right)$$ 를 의미한다.

고객 i에 대해서 비교평가 가능한 set이 있을 것이고 고객 i보다 오래 잔존한 고객 j의 생존함수를 더 크게 예측한 경우가 목적에 맞는 결과를 반환한 것이 될 것이다. C-Index 값은 0에서 1사이 값을 갖게 되며, 대소를 비교하는 것이기 때문에 random한 결과는 0.5를 반환할 것으로 기대할 수 있다. 따라서 상식적으로 모델 결과는 0.5에서 1사이의 값이 나와야하며 1에 가까울수록 모델의 성능이 더 우수한 것으로 판단할 수 있다.

변수의 영향력이 시간과 무관하다고 보는 Proportional Hazard Ratio 가정을 위배하는 변수가 포함된 경우, 적절한 조치를 취해야 한다. 그렇다면 가장 먼저 수행해야할 항목은 변수의 영향력이 시간과 무관한지를 검증할 필요가 있다. 이는 lifelines 라이브러리에서 지원하는 함수를 통해 Schoenfeld residual을 확인하는 것으로 수행 가능하다. 다음과 같이 plot을 그려주게 되는데 변수가 시간에 무관하고 parameter가 잘 추정되었다면, 모든 시간 t에서 residual값은 0 근처에서 움직여야 한다. 라이브러리의 함수는 Proportional Hazard Ratio 가정을 잘 만족하지 못하는 변수들의 Residual Plot을 보여준다.

![Cox](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/cox.png?raw=true){:width="80%" height="30%"}{: .aligncenter}

따라서 이러한 변수들을 Stratified하여 Cox Regression을 따로 fitting할 수도 있다. 이를 Stratified Cox Regression이라 부르는데 이 경우 다음과 같이 서로 다른 baseline hazard function이 범주의 개수 K개만큼 생성된다.

$$ h_{k}(t\vert X = x) = h_{0k}(t)\text{exp}(x^{T}\beta) $$

범주의 개수만큼 서로 다른 baseline hazard function을 생성하여 모델을 fitting하는 것이며 stratified 되지 않는 다른 변수의 coefficient 값은 동일하게 처리된다.

Stratified 방식이 가장 대표적인 대처 방안으로 알려져있는데 다른 대처방식도 존재한다. Cox Regression은 기본적으로 linear regression 형태를 보이는데 변수에 대한 basis spline 옵션을 통해 non-linear한 형태를 부여할 수 있다.

또 다른 방식은 Time-varying 변수에 대한 episodic data format으로 데이터 형태를 변환하여 Cox Regression을 수행하는 것이다. 기존 데이터에서는 고객 1명에 대해 1개의 row가 데이터로 존재했다면, 이를 시간 구역별로 n개로 변경하는 것이다.

각각의 row가 각 시간마다 고객의 상태(event 발생여부) 등을 표현하게 되며 이를 통해 매 시간마다 변수가 정적인 상태가 될 수 있다. event(이탈) 발생 여부는 개인별 가장 마지막 row에 반영해준다.

데이터를 episodic format으로 바꾸면 start, stop 컬럼이 생성되는데 Proportional Hazard Ratio 가정을 만족하지 않는 변수와 stop 컬럼의 값을 곱하여 interaction term을 생성하여 time varying 변수를 생성해낸다. 이후 라이브러리에서 CoxTimeVaryingFitter()함수를 통해 모델을 적합시키면 된다.

상기 내용에 관한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Cox%20PH%20Model.ipynb)에서 확인 가능합니다.


#### 참고문헌

1. [Survival Analysis](https://hyperconnect.github.io/2019/10/03/survival-analysis-part3.html)
2. [lifelines](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html)