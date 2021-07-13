---
layout: post
title:  "Gaussian Process"
date: 2021-07-11
author: seolbluewings
categories: Statistics
---

[작성중...]

Input 데이터 $$x_{i}$$에 대한 Target Output $$t_{i}$$는 일반적으로 $$t_{i}=y(x_{i})+\epsilon_{i}$$ 로 표현 가능하다. 모델은 어떠한 현상을 수식으로 표현하는 것인데 이는 결국 $$t_{i}$$를 가장 잘 설명할 수 있는 최적의 함수 $$y(x_{i})$$를 구하는 것이라고 볼 수 있다.

함수 $$y(x_{i})$$가 선형회귀 식이라고 할 때, 지금까지는 함수 $$y(x_{i})$$의 parameter인 $$\theta$$의 분포 $$p(\theta\vert x,y)$$를 찾는 것을 목표로 했다. 그러나 이제는 함수 자체에 대한 추론을 해보고 싶다. 이 함수 자체에 대한 추론을 진행하는 것이 가우시안 과정(Gaussian Process)이다.

Gaussian Process는 다른 베이지안 추론과 유사하게 함수에 대한 Prior 분포를 설정하고 데이터를 통해 관찰한 뒤, 함수에 대한 Posterior 분포를 생성한다.
 
#### Linear Regression을 통한 출발

앞서 논의했던 베이지안 회귀는 벡터 $$\Phi = (1,\Mathbf{X})$$ 를 선형결합하는 것이었다. 만약 회귀식이 Polynomial일 경우는 $$\Phi$$가 $$(1,\mathbf[X},\mathbf{X}^{2},...)$$ 과 같이 확장 될 수 있다. 벡터 $$\Phi$$ 안에 속한 값들은 총 M개의 기저함수로 총 M개의 기저함수의 선형 결합으로 정의된 모델을 다음과 같이 표현할 수 있다.

$$y(\mathbf{x}) = \omega^{T}\Phi(\mathbf{x})$$

기존에는 parameter $$\omega$$ 에 대한 Prior 분포 $$ p(\omega) \sim \mathcal{N}(0,\alpha^{-1}I) $$ 를 가정했고 관찰되는 데이터 포인트를 활용해서 $$p(\omega\vert\mathbf{X})$$ 란 Posterior를 구해서 $$\omega$$의 분포를 더욱 정밀하게 만드는 추론을 진행한다.

$$\mathbf{y} = \{y(x_{1}),...,y(x_{n})\}$$ 이라할 때, $$\mathbf{y}$$ 분포는 다음과 같은 과정을 통해 구할 수 있다.

먼저 $$\mathbf{y}$$ 는 $$\omega$$의 선형 결합으로 이루어진다. $$\omega$$가 가우시안 분포이고 가우시안 분포의 선형 결합은 역시 가우시안 분포이기 때문에 $$\mathbf{y}$$ 역시 가우시안 분포를 갖는다. 따라서 $$\mathbf{y}$$의 평균과 분산만 안다면 분포를 특정지을 수 있다.

$$
\begin{align}
\mathbb{E}(\mathbf{y}) &= \Phi\mathbb{E}(\omega)=0 \\
\text{Cov}(\mathbf{y}) &= \mathbb{E}(\mathbf{y}\mathbf{y}^{T}) \\
&= \Phi\mathbf{E}(\omega\omega^{T})\Phi^{T} = \frac{1}{\alpha}\Phi\Phi^{T}
\end{align}
$$

이 때, $$ \frac{1}{\alpha}\Phi\Phi^{T} = K $$로 정의내리고 이 K를 Gram Matrix로 표현할 수 있다. 총 N개의 데이터 포인트가 있고 M차원의 기저함수가 존재한다면 Gram Matrix의 원소 $$K_{nm}$$ 는 다음과 같이 표현 가능하다.

$$ K_{nm} = k(x_{n},x_{m}) = \frac{1}{\alpha}\phi(x_{n})^{T}\phi(x_{m}) $$

결국 $$\mathbf{y}$$의 분포는 다음과 같이 표현 가능하다.

$$ P(\mathbf[y}) \sim \mathcal{N}(\mathbf{y}\vert 0,K) $$



#### 참조 문헌
1. [Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance) <br>
2. [Permutation Feature Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
