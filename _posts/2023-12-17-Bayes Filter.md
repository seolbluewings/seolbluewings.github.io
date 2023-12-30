---
layout: post
title:  "Bayes Filter"
date: 2023-12-17
author: seolbluewings
categories: Statistics
---

Bayes Filter는 Kalman Filter, Particle Filter의 기본이 되는 개념으로 Bayes Rule을 따르는 재귀적인 필터이다. Bayes Filter는 prior와 likelihood를 통해서 posterior를 구하고 이 posterior가 다음 단계의 prior로 작용한다는 점에서 재귀적이라 표현한다.

Bayes Filter에 사용되는 notation을 활용하여 표현하면 다음과 같다.

초기시간 1부터 시간 t까지 측정센서 값의 시퀀스를 $$ \mathbf{z} =\{z_{1},z_{2},...,z_{t}\} $$, 제어입력의 시퀀스를 $$ \mathbf{u} =\{u_{1},u_{2},...,u_{t}\} $$ 로 정의하고 이 값이 주어졌을 때, 이를 조건부로 갖는 상태(state) 변수 $$ \mathbf{x} =\{x_{1},x_{2},...,x_{t}\} $$ 의 conditional pdf를 구하는 것이라 할 수 있다.

$$ \text{bel}(x_{t}) = p(x_{t}\vert z_{1:t},u_{1:t}) $$

이는 1부터 t시점까지의 센서값(z), 제어입력값(u)을 이용해서 t시점의 상태값 $$x_{t}$$를 확률적으로 추정하는 작업으로 $$ \text{bel}(x_{t}) $$ 값은 시간 t에서의 belief state 값이라고 부른다.

![label](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayes_filter.png?raw=true){:width="90%" height="80%"}{: .aligncenter}

Bayes Filter의 알고리즘은 다음의 프로세스를 따른다.


> $$\text{for all} x_{t}$$ &nbsp &nbsp do :   
>> $$\overline{\text{bel}}(x_{t}) = \int p(x_{t}\vert u_{t},x_{t-1})\text{bel}(x_{t-1})dx $$   
>> $$ \text{bel}(x_{t}) = \eta p(z_{t}\vert x_{t})\overline{\text{bel}}(x_{t}) $$   

> $$\text{end for} $$   
> $$ \text{return} &nbsp &nbsp \text{bel}(x_{t}) $$   


이 Bayes Filter 프로세스는 다음과 같이 해석할 수 있다.

우선 첫번째 줄은 제어 업데이트(control update) 또는 예측(prediction)이라 불리는 단계로 제어값을 활용하여 상태값을 업데이트하는 작업이다. $$\text{bel}(x_{t-1})$$은 t-1시점의 상태값을 의미하며 $$p(x_{t}\vert u_{t},x_{t-1})$$ 값은 t-1시점의 상태값과 t시점의 제어값이 주어진 상황에서의 현재 t시점의 상태값의 확률 분포로 이 2가지 값을 활용하여 현재의 상태값을 예측하는 행위라 볼 수 있다.

두번째 줄은 측정 업데이트(measurement update) 또는 보정(correction)으로 불리며 이 작업은 센서값을 통해 측정한 값을 활용하여 prediction한 값을 보정한다.

이 2가지 제어/측정 업데이트 프로세스는 다음의 이유로 만족된다. 먼저 제어 업데이트 단계의 로직은 이러하다.

$$\overline{\text{bel}}(x_{t})$$값은 현재 t시점에 센서값 $$z_{t}$$ 없이 현재 상태를 추정한 것으로 $$\overline{\text{bel}}(x_{t}) = p(x_{t} \vert z_{1:t-1},u_{1:t})$$ 식이 성립된다.

$$
\begin{align}
\overline{\text{bel}}(x_{t}) &= p(x_{t} \vert z_{1:t-1},u_{1:t}) \\ \newline
&= \int p(x_{t}\vert x_{t-1},z_{1:t-1},u_{1:t})p(x_{t-1}\vert z_{1:t-1},u_{1:t})dx_{t-1}
\end{align}
$$

이 때, 조건부 $$x_{t-1}$$는 $$z_{1:t-1},u_{1:t-1}$$ 값을 포함한다는 Markov Assumption과 모든 $$\mathbf{u}$$와 $$\mathbf{z}$$는 독립적이라는 가정으로 인해 $$p(x_{t}\vert x_{t-1},z_{1:t-1},u_{1:t}) = p(x_{t}\vert x_{t-1},u_{t})$$ 를 만족하고 $$p(x_{t-1}\vert z_{1:t-1},u_{1:t})$$의 경우는 t-1시점에 $$u_{t}$$값이 존재할 수 없으므로 $$p(x_{t-1}\vert z_{1:t-1},u_{1:t-1})$$과 동일하며 이것은 $$\overline{\text{bel}}(x_{t-1})$$이라 표현할 수 있다. 따라서 제어 업데이트의 수식이 구성되는 것이다. 

$$\text{bel}(x_{t}) = p(x_{t}\vert z_{1:t},u_{1:t})$$를 구하는 측정 업데이트 단계의 로직은 다음과 같다. 

$$
\begin{align}
p(x_{t}\vert z_{1:t},u_{1:t}) &= \frac{p(z_{t}\vert x_{t},z_{1:t-1},u_{1:t})p(x_{t}\vert z_{1:t-1},u_{1:t})}{p(z_{t}\vert z_{1:t-1},u_{1:t})} \\ \newline
&= \eta p(z_{t}\vert x_{t},z_{1:t-1},u_{1:t})p(x_{t}\vert z_{1:t-1},u_{1:t}) \\ \newline
&= \eta p(z_{t}\vert x_{t})p(x_{t}\vert z_{1:t-1},u_{1:t})
\end{align}
$$

$$p(x_{t}\vert z_{1:t-1},u_{1:t})$$ 값은 제어 업데이트 단계에서 구하는 $$\overline{\text{bel}}(x_{t})$$ 값과 동일하며 Markov Assumption으로 인해 $$x_{t}$$가 $$z_{1:t-1},u_{1:t}$$를 포함하고 있어 2번째 줄에서 3번째 줄로 넘어갈 수 있는 것이 가능하다. 이는 현재의 위치 $$x_{t}$$에서 어떠한 물체를 관측할 때, 그 관측값에 대한 이전까지의 관측값($$z_{1:t-1}$$)과 이전까지의 제어값($$u_{1:t}$$)은 의미가 없고 오직 $$x_{t}$$ 값만이 의미가 있다는 것을 말한다.

이렇게 계산된 $$\text{bel}(x_{t})$$ 값은 그 다음 $$t+1$$시점에서 이전시점의 상태값으로 역할을 하여 재귀적인 단계가 진행된다.
다만, 제어 업데이트 단계에서 진행하는 적분으로 인해 이슈가 발생한다.

만약 $$\mathbf{x}$$가 discrete한 변수라면 이 적분은 확률값을 모두 더하는 것으로 치환될 수 있지만 discrete하지 않고 continuous하다면 적분을 해야만하는데 적분이 복잡하거나 적분이 불가능한 경우가 발생할 수도 있다. 적분 불가능 케이스에 대응하는 방법은 2가지가 있다.

첫째, 적분이 가능한 정규분포만 사용하는 것으로 이 경우 Bayes Filter를 Kalman Filter라고 부른다.
둘째, 적분 대신 Monte Carlo Integration이라는 Sampling 기반으로 근사값을 추론하게 되는데 이를 Particle Filter라고 부른다.


#### 참조 문헌
1. [베이즈 필터 (Bayes Filter)](https://gaussian37.github.io/autodrive-ose-bayes_filter/) <br>
2. [SLAM - 베이즈 필터 알고리즘 (Bayes Filter Algorithm)](https://blog.naver.com/PostView.nhn?blogId=junghs1040&logNo=222345147315)

