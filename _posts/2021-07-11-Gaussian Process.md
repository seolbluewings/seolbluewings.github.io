---
layout: post
title:  "Gaussian Process"
date: 2021-07-11
author: seolbluewings
categories: Statistics
---

Input 데이터 $$x_{i}$$에 대한 Target Output $$t_{i}$$는 일반적으로 $$t_{i}=y(x_{i})+\epsilon_{i}$$ 로 표현 가능하다. 모델은 어떠한 현상을 수식으로 표현하는 것인데 이는 결국 $$t_{i}$$를 가장 잘 설명할 수 있는 최적의 함수 $$y(x_{i})$$를 구하는 것이라고 볼 수 있다.

함수 $$y(x_{i})$$가 선형회귀 식이라고 할 때, 지금까지는 함수 $$y(x_{i})$$의 parameter인 $$\theta$$의 분포 $$p(\theta\vert x,y)$$를 찾는 것을 목표로 했다. 그러나 이제는 함수 자체에 대한 추론을 해보고 싶다. 이 함수 자체에 대한 추론을 진행하는 것이 가우시안 과정(Gaussian Process)이다.

Gaussian Process는 다른 베이지안 추론과 유사하게 함수에 대한 Prior 분포를 설정하고 데이터를 통해 관찰한 뒤, 함수에 대한 Posterior 분포를 생성한다.
 
#### Linear Regression을 통한 출발

앞서 논의했던 베이지안 회귀는 벡터 $$\Phi = (1,\mathbf{X})$$ 를 선형결합하는 것이었다. 만약 회귀식이 Polynomial일 경우는 $$\Phi$$가 (1,$$\mathbf{X},{\mathbf{X}}^{2}$$,...) 과 같이 확장 될 수 있다. 벡터 $$\Phi$$ 안에 속한 값들은 기저함수라고 부르며 총 M개의 기저함수의 선형 결합으로 정의된 모델을 다음과 같이 표현할 수 있다.

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

이 때, $$ \text{Cov}(\mathbf{y}) = K $$로 정의내리고 이 K를 Gram Matrix로 표현할 수 있다. 총 N개의 데이터 포인트가 있고 M차원의 기저함수가 존재한다면 Gram Matrix의 원소 $$K_{nm}$$ 는 다음과 같이 표현 가능하다.

$$ K_{nm} = k(x_{n},x_{m}) = \frac{1}{\alpha}\phi(x_{n})^{T}\phi(x_{m}) $$

결국 $$\mathbf{y}$$의 분포는 다음과 같이 표현 가능하다.

$$ p(\mathbf{y}) \sim \mathcal{N}(\mathbf{y}\vert 0,K) $$

#### Gaussian Process를 통한 회귀

Target Output인 $$t_{i}$$는 $$t_{i}=y(x_{i})+\epsilon_{i}$$ 과 같이 예측값과 오차의 합으로 나누어 표현할 수 있다. $$t_{i}$$는 데이터를 통해 실제 관측한 값이고 $$y(x_{i})$$는 오차없이 오로지 예측을 통해서만 구한 값이다. $$\epsilon_{i}$$는 오차항을 의미한다.

오차항이 $$\epsilon \sim \mathcal{N}(0,\beta^{-1})$$ 분포를 따른다고 할 때, $$t_{i}$$의 분포는 다음과 같이 표현 가능하다.

$$p(t_{i}\vert y_{i}) \sim \mathcal{N}(t_{i}\vert y_{i},\beta^{-1})$$

데이터가 총 N개 존재한다고 한다면, $$P(\mathbf{t}\vert\mathbf{y}) \sim \mathcal{N}(\mathbf{t}\vert\mathbf{y},\beta^{-1}I_{N})$$ 으로 표현 가능하다.

이를 $$\mathbf{t}$$에 대한 marginalized distribution을 구하면 다음과 같다.

$$ P(\mathbf{t}) = \int \mathcal{N}(\mathbf{t}\vert\mathbf{y},\beta^{-1}I_{N})\mathcal{N}(\mathbf{y}\vert 0,K)d\mathbf{y}$$

$$\mathbf{t},\mathbf{y}$$의 Joint distribution은 $$ p(\mathbf{t}\vert\mathbf{y})p(\mathbf{y}) = P(\mathbf{t},\mathbf{y}) = p(\mathbf{z}) $$ 로 표현이 가능하다. $$\mathbf{t},\mathbf{y}$$ 둘의 Joint distribution을 편하기 쓰기 위해 $$\mathbf{z}$$란 표기를 사용한다.

양변에 로그를 취하면 다음과 같이 수식 전개가 가능하다.

$$
\begin{align}
\log{p(\mathbf{z})} &= \log{p(\mathbf{y})} + \log{p(\mathbf{t}\vert\mathbf{y})} \nonumber \\
&= -\frac{1}{2}(\mathbf{y}-0)^{T}K^{-1}(\mathbf{y}-0) - \frac{1}{2}(\mathbf{t}-\mathbf{y})^{T}\beta I_{n}(\mathbf{t}-\mathbf{y}) + C \nonumber \\
&= -\frac{1}{2}\mathbf{y}^{T}K^{-1}\mathbf{y}-\frac{1}{2}(\mathbf{t}-\mathbf{y})^{T}\beta I_{n}(\mathbf{t}-\mathbf{y}) + C \nonumber \\
&= -\frac{1}{2}\mathbf{y}^{T}K^{-1}\mathbf{y} -\frac{\beta}{2}\mathbf{t}^{T}\mathbf{t}+\frac{\beta}{2}\mathbf{t}\mathbf{y}+\frac{\beta}{2}\mathbf{y}\mathbf{t}-\frac{\beta}{2}\mathbf{y}^{T}\mathbf{y} \nonumber
\end{align}
$$

이 수식은 Quadratic Form 으로 정리가 가능하며, 다음의 두가지 Matrix를 정의할 때

$$
\begin{align}
\mathbf{z} &= \begin{pmatrix} \mathbf{y} \\ \mathbf{t} \end{pmatrix}
\\
R^{-1} &=
\begin{pmatrix}
K & K\beta I_{N}(\beta I_{N})^{-1} \\
(\beta I_{N})^{-1}(\beta I_{N})K & (\beta I_{N})^{-1}+(\beta I_{N})^{-1}(\beta I_{N})K(\beta I_{N})(\beta I_{N})^{-1}
\end{pmatrix}
\end{align}
$$

Joint Distribution은 다음과 같은 Multivariate Gaussian 분포를 따른다.

$$ p(\mathbf{y},\mathbf{t}) \sim \mathcal{N}\left((\mathbf{y},\mathbf{t})\vert (0,0),\begin{pmatrix} K & K \\ K & (\beta I_{N})^{-1}+K \end{pmatrix}\right) $$

우리가 관심있는 분포는 $$p(\mathbf{t})$$이고 이는 Joint Distribution을 marginalized 하여 구할 수 있다. Multivariate Gaussian Distribution의 특징을 참고한다면 $$p(\mathbf{t})$$는 다음과 같이 구할 수 있다.

$$ p(\mathbf{t}) \sim \mathcal{N}(\mathbf{t} \vert 0,(\beta I_{N})^{-1}+K) $$

여기서 K는 Gram Matrix로 일반적으로 다음과 같이 정의한다.

$$ K_{nm} = k(x_{n},x_{m}) = \frac{1}{\alpha}\phi(x_{n})^{T}\phi(x_{m}) $$

그러나 우리가 가장 궁금해하는 것은 observed data point $$\mathbf{t} = \{t_{1},...,t_{N}\}$$ 를 이용해 새로운 $$t_{N+1}$$의 분포를 정확하게 맞추는 것이다. 그렇다면 정말 우리에게 필요한 분포는 $$p(t_{N+1}\vert \mathbf{t})$$ 가 되겠다.

이 $$p(t_{N+1}\vert \mathbf{t})$$ 분포는 풀어서 표현하면 다음과 동일하다.

$$ p(t_{N+1}\vert \mathbf{t}) = \frac{p(t_{1},....,t_{N+1})}{p(t_{1},...,t_{N})} $$

이 수식에서의 분모에 해당하는 분포는 이미 알고 있다. 따라서 우리는 분자에 해당하는 분포 $$p(\mathbf{t}_{N})$$ 만 알 수 있다면, 최종 목표인 $$p(t_{N+1}\vert \mathbf{t}_{N})$$ 을 구할 수 있다.

$$\mathbf{t}_{N+1}$$의 분포는 다음과 같다. 여기서 $$\text{Cov}_{N}$$은 $$p(\mathbf{t}$$의 Covariance인 K 이며, $$\mathbf{k}^{T} = (K_{(N+1)(1)},....K_{(N+1)(N)})$$ 을 의미한다.

$$
p(t_{1},...t_{N},t_{N+1}) \sim \mathcal{N}\left(\mathbf{t}\vert 0, \begin{pmatrix} \text{Cov}_{N} & \mathbf{k} \\ \mathbf{k}^{T} & k_{(N+1)(N+1)}+\beta \end{pmatrix}\right)
$$

Multivariate Gaussian Distribution의 특성을 다시 한 번 사용하면 $$p(t_{n+1}\vert t_{1},...,t_{n})$$ 분포는 다음과 같이 구할 수 있다.

$$
\begin{align}
p(t_{N+1}\vert t_{1},...,t_{N}) &\sim \mathcal{N}(t_{N+1}\vert 0+\mathbf{k}^{T}\text{Cov}_{N}^{-1}(\mathbf{t}_{N}-0), K_{(N+1)(N+1)}+\beta - \mathbf{k}^{T}\text{Cov}_{N}^{-1}\mathbf{k}) \nonumber \\
\mu(t_{N+1}) &= \mathbf{k}^{T}\text{Cov}_{N}^{-1}\mathbf{t}_{N} \nonumber \\
\sigma^{2}(t_{N+1}) &=  k_{(N+1)(N+1)}+\beta - \mathbf{k}^{T}\text{Cov}_{N}^{-1}\mathbf{k} \nonumber
\end{align}
$$

평균 $$\mu(t_{N+1})$$ 값은 $$t_{N+1}$$이 관측될 가능성이 가장 높은 예측 포인트가 될 것이다.

이 과정이 Gaussian Process를 이용해 회귀분석을 정의하는 핵심적인 맥락이다. $$\mathbf{k}$$는 input 데이터 $$\mathbf{X}$$에 의해 정해지는 Kernel Function이고 기존에 회귀분석을 했을 때 추정했던 weight $$\omega$$는 이제 Kernel Function 속에 들어가 있는 것으로 보면 된다.

Kernel Function을 통해서 데이터 포인트 간의 관계에 대한 설정이 가능하며 기존 회귀분석이 $$\omega$$에 대한 명확한 값 또는 분포를 계산하여 $$\hat{t}_{N+1}$$의 값을 하나의 포인트로 특정했던 것과 달리 Gaussian Process를 통한 회귀는 $$t_{N+1}$$에 대한 Posterior Predictive Distribution을 구한다.


#### 참조 문헌
1. [Gaussian Process](https://kaist.edwith.org/aiml-adv/lecture/21300) <br>
2. [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/#Posterior)
3. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)