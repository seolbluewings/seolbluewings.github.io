---
layout: post
title:  "편향-분산 트레이드오프(Bias Variance Trade Off)"
date: 2020-05-17
author: YoungHwan Seol
categories: Statistics
---

우리는 모델을 통한 예측을 진행할 때, 모델의 퍼포먼스를 측정하게 된다. 분류의 문제라면 정확도(accuracy)나 F1 score 같은 것들을 이용할 것이고 회귀분석의 경우라면 평균제곱오차(MSE)가 모델의 퍼포먼스를 측정하는데 가장 빈번하게 사용하는 기준이 된다.

결과값 $$\bf{y}$$에 대한 예측을 한다고 할 때, MSE는 다음과 같이 편향(이하 bias)의 제곱과 분산의 합으로 표현될 수 있다.

$$
\begin{align}
\text{MSE}(\bf{\hat{y}}) &= \mathbb{E}(\hat{\bf{y}}-\bf{y})^{2} = \mathbb{E}(\hat{\bf{y}}-\mathbb{E}(\hat{\bf{y}})+\mathbb{E}(\hat{\bf{y}})-\bf{y})^{2} \nonumber \\
&= (\hat{\bf{y}}-\mathbb{E}(\hat{\bf{y}}))^{2} + \mathbb{E}(\mathbb{E}(\hat{\bf{y}})-\bf{y})^{2} \nonumber \\
&= \text{Var}(\hat{\bf{y}}) + \text{bias}^{2} \nonumber
\end{align}
$$

#### 분산이 의미하는 바

$$(\hat{\bf{y}}-\mathbb{E}(\hat{\bf{y}}))^{2}$$ 로 표현되는 식을 통해서 알 수 있듯이 모델을 통해 예측한 $$\hat{\bf{y}}$$이 예측값의 평균 $$\mathbb{E}(\hat{\bf{y}})$$ 를 중심으로해서 어느 정도 퍼져있는지를 보여주는 수치이다. 즉, 모델에 의해 산출되는 값인 $$\hat{\bf{y}}$$이 어느 정도의 변동성을 갖는지를 보여주는 값이라 할 수 있다.

#### bias가 의미하는 바


