---
layout: post
title:  "Accelerated Failure Time Model"
date: 2021-08-30
author: seolbluewings
categories: Statistics
---

[작성중...]

생존 분석의 목표는 사건이 발생하기까지의 시간을 예측하고 더불어 생존 확률을 추정하는 것에 있다. 기존의 회귀, 분류 문제에서 사용하는 알고리즘을 생존 데이터에 바로 적용하기 어려운 것은 생존 데이터에 중도 절단이란 개념이 포함되어 있기 때문이다. 중도절단과 사건발생 시간을 고려하지 않는 모델은 잘못된 결론을 도출할 수 있다.

대표적인 생존분석 모델 중 하나가 AFT(Accelerated Failure Time) 모델이다. 이 모델은 생존분석에서 가장 널리 알려져있는 Cox PH 모델과 달리 parametric한 방식이다. 대신 parametric한 방법이기 때문에 생존시간 t에 대한 분포 가정이 필요하다는 것이 대표적인 차이라고 볼 수 있다.

또한 Cox PH 모델에서는 위험률이 설명변수의 함수로 표현되는데 AFT 모델에서는 설명변수가 생존시간 자체에 영향을 미치게 된다.

2가지 집단이 어떠한 trigger 실행여부에 따라 구분되어 있다고 가정하자. trigger가 실행되지 않은 집단을 1번 집단, trigger가 실행된 집단을 2번 집단이라 표현하고 각 집단에 대한 생존함수 간의 관계를 만족시키는 상수 $$\gamma$$ 가 존재한다고 가정하자.

$$ S_{1}(t) = S_{2}(\gamma t) \quad \forall t \geq 0 $$

이 때, $$\gamma$$는 가속화 인수(acceleration factor)라고 부르며, 이러한 가정에서 집단 1의 이탈 증가율이 집단 2의 이탈 증가율보다 $$\gamma$$배만큼 크다고 볼 수 있다.

i번째 개체에 대해 p개의 설명변수 $$\mathbf{x}_{i} = (x_{i1},...,x_{ip})$$와 생존시간 $$t_{i}$$가 관측되었다고 하자. AFT 모델은 다음과 같은 선형회귀모형을 정의하는 것에서 출발한다.

$$\log(t_{i}) = \beta_{0}+\beta_{1}x_{i1}+\cdot\cdot\cdot+\beta_{p}x_{ip}+\sigma\epsilon_{i} $$

이러한 선형회귀모형을 통해 생존시간 자체를 설명변수의 함수로 정의내리며 결국 생존시간 $$t_{i}$$는 i번째 개체의 설명변수 값들로 인해 결정된다.

개체 1과 개체2의 설명변수 중 k번째 값만 다르다고 가정하자. 개체 1은 $$x_{1k}$$ 이며 개체2의 경우는 $$x_{1k}+1$$의 값을 갖는다고 하자.

앞서 정의한 생존시간 $$t$$에 대한 선형회귀모형을 적용하면 다음과 같이 작성할 수 있을 것이다.

$$
\begin{align}
t_{1} &= \text{exp}(\beta_{0}+....+\beta_{k}x_{k}+...+\beta_{p}x_{p}+\sigma\epsilon_{1}) = \gamma_{1}\text{exp}(\sigma\epsilon_{1}) \nonumber \\
t_{2} &= \text{exp}(\beta_{0}+....+\beta_{k}(x_{k}+1)+...+\beta_{p}x_{p}+\sigma\epsilon_{2}) = \gamma_{2}\text{exp}(\sigma\epsilon_{2}) = \gamma_{1}\text{exp}(\beta_{k}+\sigma\epsilon_{2}) \nonumber
\end{align}
$$

$$\epsilon_{1}$$ 과 $$\epsilon_{2}$$ 가 동일 분포를 따른다고 가정하면, 각각의 생존함수는 다음과 같이 표현 가능할 것이다.

$$
\begin{align}
S_{1}(t) &= p(t_{1} \geq t) = p(\gamma_{1}\text{exp}(\sigma\epsilon_{1}) \geq t) = p(\text{exp}(\sigma\epsilon_{1}) \geq \gamma_{1}^{-1}t) \nonumber \\
S_{2}(t) &= p(t_{2} \geq t) = p(\gamma_{2}\text{exp}(\sigma\epsilon_{2}) \geq t) = p(\text{exp}(\sigma\epsilon_{2}) \geq \gamma_{2}^{-1}t) \nonumber \\
S_{2}(\text{exp}(\beta_{k})t) &= p(\text{exp}(\sigma\epsilon_{2}) \geq \gamma_{2}^{-1}\text{exp}(\beta_{k})t) \nonumber \\
&= p(\text{exp}(\sigma\epsilon_{2}) \geq \gamma_{1}^{-1}t) = p(\text{exp}(\sigma\epsilon_{1}) \geq \gamma_{1}^{-1}t) = S_{1}(t) \nonumber
\end{align}
$$

이 때, $$\beta_{k} > 0$$ 일 경우, 설명변수의 1단위 증가는 생존율을 증가시키고 $$\beta_{k} < 0$$ 인 경우에는 설명변수의 1단위 증가가 생존율을 감소시킨다고 볼수 있다.

설명변수의 효과가 생존시간에 비례하며 이는 곧 설명변수가 사건발생(이탈)시간을 가속시키거나 지연시키는 효과를 불러온다고 볼 수 있다.

생존함수에 적용되는 수식을 위험함수로 확장한다면 위험함수는 다음과 같이 표현될 수 있다.

$$ h_{1}(t) = \text{exp}\left(\mathbf{x}\beta\right)h_{2}\left(\mathbf{x}\beta t\right) $$

#### Weibull Accelerated Faiure Time Model

앞서 AFT 모델은 생존시간 $$t$$에 대한 분포가정이 필요하다고 언급하였다. Weibull AFT 모델은 생존시간 $$t$$가 Weibull 분포를 따르는 것을 가정한다. Weibull 분포는 부품수명 추정에 자주 사용되는 분포로 시간에 따라 고장날 확률이 높아지는 경우와 낮아지는 경우를 포함하는 분포이다. 고객의 이탈을 예측하는 과정에서도 고객의 이탈 가능성이 증가하는 구간이 충분히 존재 가능하여 Weibull 분포를 적용하기로 한다.

특히 Weibull 분포의 pdf는 2가지 term의 곱으로 이루어져있는데 $$\kappa\lambda(\lambda t)^{\kappa-1}$$ 파트를 위험함수로 $$\text{exp}(-\lambda t)^{\kappa}$$ 파트를 생존함수로 간주 가능한 특징이 있다.

$$
\begin{align}
t &\sim \text{Weibull}(\kappa, \lambda) \nonumber \\
f(t) &= \kappa\lambda(\lambda t)^{\kappa-1}\text{exp}(-\lambda t)^{\kappa} \nonumber \\
S(t) &= \text{exp}(-\lambda t)^{\kappa} \nonumber \\
h(t) &= \kappa\lambda(\lambda t)^{\kappa-1} \nonumber \\
H(t) &= (\lambda t)^{\kappa}
\end{align}
$$

위험함수 $$h(t)$$에 log를 취하면 다음과 같이 표현 가능하다. 이는 $$log{t}$$ 에 대해 기울기 $$(\kappa-1)$$과 절편 $$\log{\kappa} + \kappa\log{\lambda}$$ 를 갖는 것으로 볼 수 있다.

$$\log{h(t)} = \log{\kappa} + \kappa\log{\lambda} + (\kappa-1)\log{t}$$

따라서 $$\kappa > 1$$인 경우 시간에 따라 위험함수가 증가하며 반대로 $$\kappa < 1$$인 경우에는 시간에 따라 위험함수가 감소한다고 볼 수 있다. 

#### 참조 문헌
1. R을 이용한 생존분석 기초
2. [생존 분석(Survival Analysis) 탐구 1편](https://velog.io/@jeromecheon/%EC%83%9D%EC%A1%B4-%EB%B6%84%EC%84%9D-Survival-Analysis-%ED%83%90%EA%B5%AC-1%ED%8E%B8)
3. [The basics of survival analysis](https://sakai.unc.edu/access/content/group/2842013b-58f5-4453-aa8d-3e01bacbfc3d/public/Ecol562_Spring2012/docs/lectures/lecture27.htm)

