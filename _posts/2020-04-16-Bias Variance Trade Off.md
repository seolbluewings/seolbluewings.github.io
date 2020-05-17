---
layout: post
title:  "편향-분산 트레이드오프(Bias Variance Trade Off)"
date: 2020-04-16
author: YoungHwan Seol
categories: Statistics
---

우리는 모델을 통한 예측을 진행할 때, 모델의 퍼포먼스를 측정하게 된다. 분류의 문제라면 정확도(accuracy)나 F1 score 같은 것들을 이용할 것이고 회귀분석의 경우라면 평균제곱오차(MSE)가 모델의 퍼포먼스를 측정하는데 가장 빈번하게 사용하는 기준이 된다.

결과값 $$\bf{y}$$에 대한 예측을 한다고 할 때, MSE는 다음과 같이 편향(이하 bias)의 제곱과 분산의 합으로 표현될 수 있다.

$$
\begin{align}
\text{MSE}(\bf{\hat{y}}) &= \mathbb{E}(\hat{\bf{y}}-\bf{y})^{2} = \mathbb{E}(\hat{\bf{y}}-\mathbb{E}(\hat{\bf{y}})+\mathbb{E}(\hat{\bf{y}})-\bf{y})^{2} \nonumber \\
&= \mathbb{E}(\hat{\bf{y}}-\mathbb{E}(\hat{\bf{y}}))^{2} + (\mathbb{E}(\hat{\bf{y}})-\bf{y})^{2} \nonumber \\
&= \text{Var}(\hat{\bf{y}}) + \text{bias}^{2} \nonumber
\end{align}
$$

#### 분산이 의미하는 바

$$(\hat{\bf{y}}-\mathbb{E}(\hat{\bf{y}}))^{2}$$ 로 표현되는 식을 통해서 알 수 있듯이 모델을 통해 예측한 $$\hat{\bf{y}}$$이 예측값의 평균 $$\mathbb{E}(\hat{\bf{y}})$$ 를 중심으로해서 어느 정도 퍼져있는지를 보여주는 수치이다. 즉, 모델에 의해 산출되는 값인 $$\hat{\bf{y}}$$이 어느 정도의 변동성을 갖는지를 보여주는 값이라 할 수 있다. 분산이 큰 모델은 훈련 데이터에 높은 초점을 두고 만들어져 아직 관측하지 못한 데이터에 대해서는 일반화시키기 어려운 모델이라 할 수 있다. 따라서 분산이 큰 모델은 훈련 데이터에 굉장히 높은 성능을 보이지만, 테스트 데이터에서는 큰 오류를 낼 수 있다.

#### bias가 의미하는 바

bias는 모델을 통해 추정한 값의 평균 $$\mathbb{E}(\hat{\bf{y}})$$와 우리가 예측하고자 했던 실제값인 $$\bf{y}$$의 차이를 보여주는 값이다. 모델을 통해 맞추지 못하는 차이를 의미하는 값이라고 할 수 있으며 평균적으로 우리가 얼마나 실제값을 맞추지 못하는지 보여준다고 생각할 수 있다. bias가 큰 모델은 모델이 너무 간단한 나머지 훈련 데이터 뿐만 아니라 테스트 데이터에서도 큰 오류를 유발시킨다.

#### 그림으로 이해하기

![biasvariance](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bias_variance_tradeoff.PNG?raw=true)

이 그림은 bias와 variance를 설명할 때 활용되는 가장 유명한 그림이다. bias의 크고 작음, variance의 크고 작음에 따른 결과가 아주 직관적으로 잘 표현되어 있다. 표적지의 중앙에 점이 찍힌다는 것은 모델이 실제값을 정확하게 예측한다는 것을 의미한다.

bias가 크다는 것은 실제값과 예측값의 오차가 크게 발생하는 것을 의미하며 variance가 크다는 것은 예측값이 갖는 범위가 넓다는 것을 말한다. bias가 높다는 것은 예측과정에서 꾸준히 값을 틀리고 있다는 것이며 variance가 크다는 것은 예측값이 갖는 범위가 커져 결과의 안정성이 떨어지는 것이라 할 수 있다.


#### Bias-Variance Trade-off 관계

최고의 결과는 bias도 작고 variance도 작은 모델을 생성하는 것이다. 그러나 두가지 사항은 아래의 그림과 같이 서로 Trade-off 관계를 갖는다. X축은 모델의 복잡도를 나타내는 항목으로 모델이 복잡해질수록 bias가 줄어드나 variance가 높아지며 반대로 모델의 복잡도가 낮아질수록 bias는 증가하나 variance가 줄어드는 것을 볼 수 있다. 각각 1. 모델이 너무 복잡해서 예측값 $$\hat{\bf{y}}$$이 크게 흔들리는 것이며 2. 모델이 너무 단순해서 정확한 예측을 못하는 경우라 할 수 있다.

![biasvariance](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/model_complexity.PNG?raw=true)

이처럼 Bias를 작게하려 하니 Variance가 높아지고 Variance를 작게 만드려고 하니 Bias가 커지는 상황이 발생하게 된다. 결국 Bias와 Variance 두가지를 동시에 작아지게 만드는 것은 불가능하다. 따라서 우리는 그림에서 점선으로 표현된 최적의 복잡도를 가진 모델을 찾아야하며 이 과정은 이후 포스팅을 통해 알아보고자 한다.