---
layout: post
title:  "Granger Causality"
date: 2022-04-24
author: seolbluewings
categories: Statistics
---

[작성중...]

서로 시차가 존재하는 데이터간의 선/후행을 따져 인과관계를 알아보고자 할 때, 사용할 수 있는 방법 중 하나가 Granger Causality이다.

다음과 같이 시계열 데이터 $$\{x_{t}\}_{t=1}^{T}$$와 $$\{y_{t}\}^{T}_{t=1}$$ 가 존재할 때, $$y_{t}$$가 $$x_{t}$$의 과거 데이터 linear regression 형태로 적합되며 이 linear regression이 통계적으로 유의미할 때, $$\{x_{t}\}_{t=1}^{T}$$와 $$\{y_{t}\}^{T}_{t=1}$$는 Granger Causality 관계에 있다고 말한다.


여기서 그냥 Causality가 아닌 Granger Causality 라고 부르는 것을 주목해야한다. 결과 해석 시, $$x,y$$가 Granger Causality 관계에 있다면, 이를 $$x$$가 $$y$$의 원인(cause)이라 말하기보단 $$x$$는 $$y$$를 예측(forecase)한다고 말하는게 맞는 것으로 알려진다.

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

1. [Granger Causality](https://en.wikipedia.org/wiki/Granger_causality)
2. [그레인저 인과관계-Granger Causality](https://intothedata.com/02.scholar_category/timeseries_analysis/granger_causality/)