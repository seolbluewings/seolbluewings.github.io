---
layout: post
title:  "Cox Proportional-Hazard Model"
date: 2022-05-01
author: seolbluewings
categories: Statistics
---

[작성중...]

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


[to be continued...
]

![Granger](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Granger_Causality.png?raw=true){:width="80%" height="30%"}{: .aligncenter}

그림과 같이 데이터 $$x,y$$가 존재하고 $$x$$가 $$y$$에 대한 Granger-Cause 관계라면, 빨간색 선을 따라 확인할 수 있듯이 $$x$$에서 발견되는 패턴이 대략적으로 몇개의 lag 후에 $$y$$에서 반복된다. 따라서 과거의 $$x$$ 데이터가 $$y$$의 향후 데이터를 예측(prediction)하는데 사용된다고 말할 수 있다.

구체적인 Granger Causality 검증(test) 방안은 다음과 같다.

먼저 두개 이상의 시계열 데이터가 정상 시계열인지 확인해야 한다. 만약 시계열 데이터가 비정상 시계열이라면, 1차 이상의 차분을 수행한다. 정상 시계열은 Granger Causality 검증을 위한 전제 조건이다. 

'검증'인 만큼 통계에서 자주 사용되는 null hypothesis, alternative hypothesis가 셋팅되어야 한다.

null hypothesis는 $$x$$가 $$y$$에 대한 Granger-Cause가 없다는 것을 반영하는 수식이어야 한다. 따라서 $$y$$에 대한 예측은 $$y$$의 과거 데이터로 기반한 autoregression 형태로 표현되어야할 것이다.

$$ y_{t} = a_{0}+a_{1}y_{t-1}+\cdots+a_{k}y_{t-k} + e_{t} $$

alternative hypothesis는 $$x$$에 대한 과거 데이터가 수식에 추가되어야할 것이다.

$$ y_{t} = a_{0}+a_{1}y_{t-1}+\cdots+a_{k}y_{t-k} + b_{p}x_{t-p} + \cdots +b_{q}x_{t-q} + e_{t}  $$

검증은 일반적인 linear regression 처럼 유의성 검정(F-test)을 진행하게 되며, $$x$$의 과거 시점 데이터가 단 하나도 수식에 포함되지 않는 경우 $$x,y$$가 Granger-Cause 관계가 아니라는 null-hypothesis를 채택한다. 즉 $$y$$를 예측하는데 있어서 $$x$$의 여러 lag 데이터들 중 적어도 하나가 $$y$$를 예측하는데 유의(significant)해야한다는 것이다.

테스트 방향은 $$x \to y$$ 방향의 인과영향, $$y \to x$$ 방향의 인과영향을 체크하는 것으로 2번 수행하는 것이 이치에 맞지만, 논리적으로 확실하게 한쪽 방향의 인과성은 말이 안되는 경우 생략이 가능하다.
 
#### 참고문헌

1. [Survival Analysis](https://hyperconnect.github.io/2019/10/03/survival-analysis-part3.html)
2. [그레인저 인과관계-Granger Causality](https://intothedata.com/02.scholar_category/timeseries_analysis/granger_causality/)