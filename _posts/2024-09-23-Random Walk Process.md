---
layout: post
title:  "Random Walk Process"
date: 2024-09-23
author: seolbluewings
categories: Statistics
---

시계열 데이터(time series) 예측은 일반적인 회귀분석을 활용한 예측과 다른 부분이 있어 별도 게시글을 작성하여 이에 관한 내용을 정리하고자 한다.
시계열 데이터를 활용한 예측과 회귀분석을 통한 예측 모두 과거의 데이터를 활용하여 앞으로의 데이터를 예측하고자하는 목표는 동일하나 다음의 포인트에서 크게 다르다고 볼 수 있다.

1. 시계열 데이터는 예측 과정에서 데이터의 순서를 고려해야 한다. 일반적인 회귀분석/머신러닝 문제에서는 주로 input 데이터와 output 데이터 간의 상관관계를 파악하는 것이 주된 관심사이기 때문에 데이터 간의 순서를 중요하지 않게 생각하는 경우가 많지만 시계열 데이터를 모델링하는 과정에서는 전체 과정에서 데이터 순서를 동일하게 유지해야 한다.
2. 시계열 데이터는 데이터 기록시점(timestamp)와 그 시점의 관측값만 존재하는 경우가 있는데 시계열 데이터 그 자체의 특성만을 이용해서도 예측이 가능하다. 

시계열 데이터 관련된 시리즈를 시작하기 위해서 가장 단순한 형태의 시계열 데이터인 확률 보행(Random Walk Process)에 대해서 정리할 필요가 있다.
확률 보행에 대한 내용을 정리하면서 시계열 분석에서 가장 기본적인 개념들을 정리하고자 한다.

우선 확률 보행에 대해서 정의를 하면, <strong> 1차 차분이 정상적(stationary)이고 전후 데이터 간의 상관관계가 없는 시계열 </strong> 로 데이터가 무작위로 변화하여 시계열 데이터가 향후 무작위로 상승/하락 발생 확률이 동일한 프로세스를 의미한다. 이를 수식으로 표현하면 아래와 같다.

$$ y_{t} = y_{t-1} + C + \epsilon_{t}, \quad \epsilon_{t} \sim \mathcal{N}(0,1) $$

현재시점의 값($$y_{t}$$)은 이전 시점의 값($$y_{t-1}$$)과 난수($$\epsilon_{t}$$), 상수 $$C$$ 의 선형 결합이다.

![random_walk](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/random_walk_1.png?raw=true){:width="90%" height="80%"}{: .aligncenter}

위의 그림과 같은 시계열 데이터는 데이터가 증가/감소를 무작위로 반복하는데 이렇게 데이터의 증감이 무작위로 발생하는 경우, 이 데이터가 확률 보행이라고 볼 수 있다. 

실제 데이터를 다루는 상황에서는 주어진 데이터가 확률 보행인지 아닌지를 눈이 아닌 정량적 지표를 기반으로 식별하는 방법이 필요한데 확률 보행의 정의를 보면 차분/정상성/상관관계에 대한 정의를 내려야한다.

차분은 시점 t와 시점 t-1의 데이터 값의 변화를 계산하는 것으로 $$ y_{t}^{'} = y_{t}-y_{t-1} $$ 의 계산을 통해 생성된 데이터를 의미한다. 이 계산을 1번 수행하는 것을 1차 차분된 데이터라고 하며 n번 시행하면 n차 차분한 데이터라고 말한다. 차분된 데이터는 두 시점간의 데이터 차이를 계산하여 생성된 데이터이므로 데이터 개수의 손실이 발생한다. 차분을 진행하는 이유는 이 작업을 통해서 시계열 데이터의 평균값을 안정시킬 수 있기 때문이다.

![random_walk](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/random_walk_3.png?raw=true){:width="90%" height="80%"}{: .aligncenter}

이 그림은 앞서 그린 확률 보행 데이터를 1차 차분한 것으로 차분된 시계열 데이터가 평균,분산이 이전의 원본 데이터에 비해서 안정적으로 변한 것을 확인할 수 있다.

또한 정상성(stationary)에 대해서 알아보아야 한다. <strong> 정상성은 시계열 데이터가 시간이 지나도 데이터의 통계적 특성(시계열 데이터의 평균/분산/자기상관관계)가 변하지 않는 프로세스임을 의미한다. </strong> 정상성은 예측을 수행하는 과정을 비교적 쉽게 만드는 가정으로 다수의 시계열 데이터는 정상성을 갖지 않는다고 봐야한다. 기본적인 예측 모델인 MA(이동평균),AR(자기회귀) 모델도 데이터가 정상 시계열 조건을 만족할 때 사용가능하다. 

주어진 데이터에 대해 정상성 여부에 대한 검증은 ADF Test를 진행하여 확인 가능하다. ADF 테스트는 시계열에 단위근(unit root)이 존재하여 주어진 시계열이 비정상적이라고 보는 귀무가설($$H_{0}$$)에 대한 검증을 진행한다. 이 테스트 결과의 p-value가 0.05 미만이어야 귀무가설을 기각할 수 있다. 시계열 데이터의 정상성을 검증하는 python 코드는 아래와 같다.

```
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(random_walk)
print(f"== ADF 통계량 : {adf_result[0]}")
print(f"== p-value : {adf_result[1]}")
```

자기상관관계(Autocorrelation)는 시계열 데이터의 선행값($$y_{t-1}$$)과 후행값($$y_{t}$$) 사이의 선형 관계를 측정한 결과를 의미한다. 시계열 데이터의 자기상관관계 존재 여부에 대해서는 일반적으로 자기상관함수(ACF)를 그려 시점 차이(lag)마다의 상관관계 값의 변화를 본다. 일반적으로 lag가 커지면서 자기상관관계 값이 선형적으로 감소하는데 lag가 작은 숫자인 부분에서 자기상관관계가 0에 가까운 값을 가져야 한다. 

![random_walk](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/random_walk_2.png?raw=true){:width="90%" height="80%"}{: .aligncenter}

위 ACF plot은 각 lag마다의 자기상관계수를 표현하는데 음영처리 된 부분은 해당 lag에서의 자기상관계수값에 대한 confidence interval로 이 음영 영역 내에 값이 존재하는 경우, 해당 lag에서의 값을 0으로 간주할 수 있다. 이 그래프를 통해서 자기상관관계가 없다고 판단 된다면, 이 데이터가 확률 보행 데이터라고 볼 수 있다.

1차 차분된 데이터가 정상 시계열이면서 자기상관관계가 없다면, 이 시계열 데이터를 확률 보행이라 볼 수 있는데 이러한 시계열 데이터는 확률 보행의 특성상 데이터가 무작위로 변화하기 때문에 통계적 기법, 딥러닝 등을 활용하는 것이 무의미하다고 볼 수 있다. 


확률 보행에 대한 간략한 python 코드 예시는 다음의 [링크](https://github.com/seolbluewings/python_study/blob/master/01.study/random_walk.py)에서 확인할 수 있다.


#### 참조 문헌
1. [Time Series Forecasting in Python, 파이썬 시계열 예측 분석](https://product.kyobobook.co.kr/detail/S000213799852a) <br/>
