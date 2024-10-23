---
layout: post
title:  "Moving Average Process"
date: 2024-10-02
author: seolbluewings
categories: Statistics
---

[작성중...]

앞서 Random Walk Prcoess에 대해서 알아보았는데 Random Walk Process는 1차 차분한 시계열 데이터가 정상 프로세스이면서 자기상관관계가 없는 시계열 데이터였다. 그러나 정상 시계열 중에서도 자기상관관계가 있는 경우가 있다. 이러한 시계열 데이터는 이동평균 MA(q) 모델, 자기회귀 AR(p) 모델, 자기회귀이동평균 ARMA(p,q) 모델을 활용하여 향후 시계열 데이터의 흐름을 예측한다. 

이 모델들은 모두 과거 시계열 데이터의 선형 결합을 기반으로 미래 값을 예측하는 방식이기 때문에, 선형 시계열 모형(Linear Time Series Model)이라 표현한다. 가장 먼저 MA(q) 모델에 대해서 정리하면 내용은 다음과 같다.

<strong> 이동평균(Moving Average Process) </strong>
MA(q) 모델은 현재 시점의 값이 현재의 값과 과거 오차의 선형결합으로 표현 가능하여 다음과 같이 표기할 수 있다.

$$
y_{t} = \mu + \epsilon_{t} + \theta_{1}\epsilon_{t-1} + \cdots + \theta_{q}\epsilon_{t-q}
$$

MA(1) 모델의 경우 $$y_{t} = \mu + \epsilon_{t}+\theta_{1}\epsilon_{t-1}$$ 형태로 표현되며, MA(2) 모델은 $$y_{t} = \mu + \epsilon_{t}+\theta_{1}\epsilon_{t-1} + \epsilon_{2}\epsilon_{t-2}$$ 로 표현 가능하다.

따라서 MA(q) 모델에서 $$q$$값은 모델에 반영할 과거 오차항의 갯수를 결정짓는 값이다. q가 커질수록 더 많은 과거의 오차항이 현재값에 영향을 미친다고 볼 수 있고 MA(q) 모델을 적합하기 위해서는 q값을 결정짓는 방법을 정의해야할 것이다.

주어진 데이터에 대해서 자기상관함수를 그리고 q시점 이후로 지속적으로 자기상관계수가 유의하지 않다면, MA(q) 프로세스로 정의한다. 

실제 데이터를 바탕으로 MA(q) 프로세스를 적합하는 과정을 살펴본다면 다음과 같다. 

![ma process](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/MA1.png?raw=true){:width="60%" height="60%"}{: .aligncenter}

이 그래프는 판매량에 대한 가상 데이터인데 추세가 상승과 하강을 반복하는 것을 확인할 수 있다. 시간에 흐름에 따라 전체적으로 상승하는 모습도 있어 이 시계열 데이터가 정상성을 갖는다고 보기 어려워 보인다. 실제로 이 데이터에 대해서 ADF Test를 진행하더라도 시계열이 정상성을 갖지 않는다고 결론이 닌다.

그렇다면 이 데이터에 대해서 차분을 진행하고 차분하여 생성한 새로운 시계열 데이터에 대해서 자기상관함수를 그려본다. lag별로 자기상관계수 값을 살펴보면, lag=2 까지의 자기상관계수는 유의하다고 볼 수 있어 원본 시계열이 MA(2) Process를 따른다고 판단할 수 있다. 

![ma process](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/MA2.png?raw=true){:width="60%" height="60%"}{: .aligncenter}

MA Process의 차수 q를 식별했다면, MA(q) 모델을 훈련 데이터에 fitting하여 향후 값을 예측할 수 있다. 단, 여기서 주의해야할 점은 MA(q) 모델이 시계열 데이터가 정상성을 갖는걸 전제한 상태에서 출발한다는 것이다. 
따라서 차분하여 정상성을 갖는 시계열에 대해서 MA(q) 모델을 fitting하고 그 결과를 다시 원복시키는 과정을 거쳐 원래 데이터에 대한 예측을 진행해야 한다.

추가로 주의해야할 점은 MA(q) 모델은 한번에 q단계 이상의 데이터를 예측할 수 없다는 특징을 갖는다. MA(q) 모델의 정의에서 확인할 수 있듯이 MA(q) 모델은 과거 q개의 오차항의 선형 결합으로 이루어져있다.
따라서 예측을 한다고 하면, MA(q) 모델은 앞으로의 q단계까지만 예측이 가능하다. q+1시점부터는 모델이 과거의 오차항을 활용하지 않고 오로지 평균으로만 예측하기 때문에 평균값으로만 예측을 진행할 것이다. 이러한 예측은 유의하지 않다.

MA(q) 모델의 특성상, MA(q) 모델을 적합하기 위해서는 롤링 예측(rolling forecast)을 진행해야 한다. 1번 혹은 q번의 시간 단계씩 반복해서 예측하여 목표 시점(t)까지 예측을 진행한다.
이러한 방식의 예측을 수행하려면, 전체 테스트 데이터에 대한 예측을 완료할 때까지 반복적으로 fitting을 진행하여 예측을 해야한다. 이러한 함수는 다음과 같이 코드를 정의할 수 있다.

```
def rolling_forecast(df:pd.DataFrame, tr_length : int, horizon : int, window : int, method : str):
    tot_length = tr_length + horizon

    if method == 'MA':
        MA_pred_list = []
        
        for i in range(tr_length, tot_length, window):
            model = SARIMAX(df.iloc[:i], order = (0,0,2))
            res   = model.fit(disp = False)
            predictions = res.get_prediction(0, i+window-1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            MA_pred_list.extend(oos_pred)
        return MA_pred_list
```

이 rolling_forecast 함수를 적용한 뒤, 1차 차분을 다음과 같이 역변환 진행한다. $$y_{1},y_{2}$$ 값을 원복하기 위해서는 다음과 같이 차분된 값의 누적 합계를 계산한다.

$$\begin{align}
y_{1} &= y_{0}+y_{1}^{'} = y_{0} + y_{1} - y_{0} = y_{1} \\
y_{2} &= y_{0} + y_{1}^{'} + y_{2}^{'} = y_{0} + y_{1} - y_{0} + y_{2} - y_{1} = y_{2}
\end{align}
$$

MA Process에 대한 간략한 python 코드 예시는 다음의 [링크](https://github.com/seolbluewings/python_study/blob/master/01.study/ma_process.py)에서 확인할 수 있다.


#### 참조 문헌
1. [Time Series Forecasting in Python, 파이썬 시계열 예측 분석](https://product.kyobobook.co.kr/detail/S000213799852a) <br/>
