---
layout: post
title:  "Bayes by Backprop"
date: 2022-01-16
author: seolbluewings
categories: Statistics
---

[작성중...]

Bayesian Neural Network는 모델 학습 과정에서 weight $$\mathbf{w}$$가 determinstic한 값을 갖는 것으로 간주하는 기존의 [Neural Network](https://seolbluewings.github.io/statistics/2020/09/28/Neural-Network-copy.html)와 달리 weight $$\mathbf{w}$$에 대한 확률 분포를 설정함으로써 weight $$\mathbf{w}$$와 output $$\mathbf{y}$$ 에 대한 분포를 제공하여 모델의 불확실성(uncertainty) 까지도 제공하는 모델이라 할 수 있다.

그렇다면 Bayesian Neural Network(이하 BNN)는 기존의 Neural Network(이하 NN)와 무슨 포인트에서 차별성을 갖는지? 어떠한 방식으로 weight와 output에 대한 uncertainty를 도출하는지를 알아볼 필요가 있다.

![BNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/BNN1.png?raw=true){:width="70%" height="30%"}{: .aligncenter}

이 그림은 BNN을 설명하는 대표적인 이미지라 할 수 있다. 기존 NN의 weight는 -0.1,0.6과 같이 하나로 딱 떨어지는 determinstic한 값을 가지고 있었지만, BNN에서는 각 weight마다의 분포가 존재하게 된다. weight에 대한 분포라는 방식의 uncertainty가 부여가 되면 그 weight을 이용해 추정한 output 값에도 uncertainty가 부여되는 것은 지극히 합리적이다.

그래서 BNN 학습 과정에서 단 1개의 Network를 학습하는 것이 아닌 여러개의 Network를 학습하고 이를 ensemble 하는 과정이 필요해진다.

기존의 NN을 probabilistic model 형태로 표현하면 $$p(\mathbf{y}\vert \mathbf{x},\mathbf{w})$$ 로 표현할 수 있고 train data $$\mathcal{D} = \{(x_{i},y_{i})\}$$ 가 주어진 상황에서 weight $$\mathbf{w}$$에 대한 MLE를 구함으로써 $$\hat{\mathbf{w}}$$ 를 추정하게 되는데 gradient descent 등의 방식을 사용해서 구한다.

$$\mathbf{w}^{MLE} = \text{argmax}_{\mathbf{w}}\log{p(\mathcal{D}\vert\mathbf{w})} = \text{argmax}_{\mathbf{w}}\sum_{i=1}^{n}\log{p(y_{i}\vert x_{i},\mathbf{w})}$$

기존의 NN과 달리 BNN에 대한 Bayesian Inference는 train data $$\mathcal{D}$$가 주어진 상황에서 $$\mathbf{w}$$에 대한 posterior distribution을 구함으로써 달성할 수 있다. $$\mathbf{w}$$ 에 대한 posterior를 구함으로써 새로운 데이터 $$x_{new}$$ 에 대한 $$y_{new}$$의 label을 예측할 수 있게 된다.

$$ p(y_{new}\vert x_{new}) = \mathbb{E}_{p(\mathbf{w}\vert\mathcal{D})}[p(y_{new}\vert x_{new},\mathbf{w})] $$

그러나 모든 Bayesian 문제에서 발생하는 문제는 posterior $$p(\mathbf{w}\vert\mathcal{D})$$ 의 분포를 정확하게 알아내기 어렵다는 것이다. 그래서 이 posterior distribution에 대한 variational approximation을 구해야하는 문제로 상황이 바뀌게 된다.

$$p(\mathbf{w}\vert\mathcal{D})$$에 가장 잘 근사하는 Variational 분포 $$q(\mathbf{w}\vert\theta)$$를 구해야하는데 이는 다음의 수식을 만족하는 $$\theta^{*}$$를 찾는 것과 동등하다.

$$
\begin{align}
\theta^{*} &= \text{argmin}_{\theta}KL\left[q(\mathbf{w}\vert\theta)\vert p(\mathbf{w}\vert\mathcal{D})  \right] \nonumber \\
&= \text{argmin}_{\theta}\int q(\mathbf{w}\vert\theta)\log{\frac{q(\mathbf{w}\vert\theta)}{p(\mathbf{w})p(\mathcal{D}\vert\mathbf{w})}}d\mathbf{w} \nonumber \\
&= \text{argmin}_{\theta}KL\left[q(\mathbf{w}\vert\theta)\vert p(\mathbf{w})  \right] - \mathbb{E}_{q(\mathbf{w}\vert\theta)}\left[\log{p(\mathcal{D}\vert\mathbf{w})}\right] \nonumber
\end{align}
$$

이 수식에서 $$KL\left[q(\mathbf{w}\vert\theta)\vert p(\mathbf{w})  \right] - \mathbb{E}_{q(\mathbf{w}\vert\theta)}\left[\log{p(\mathcal{D}\vert\mathbf{w})}\right]$$ 만 따로 떼어내서 볼 필요가 있다.

$$
\begin{align}
#\mathcal{F}(\mathcal{D},\theta) &= KL\left[q(\mathbf{w}\vert\theta)\vert p(\mathbf{w})  \right] - \mathbb{E}_{q(\mathbf{w}\vert\theta)}\left[\log{p(\mathcal{D}\vert\mathbf{w})}\right] \nonumber \\
&= \int q(\mathbf{w}\vert\theta}\log{\frac{q(\mathbf{w}\vert\theta)}{p(\mathbf{w})}}d\mathbf{w} - \int q(\mathbf{w}\vert\theta)\log{p(\mathcal{D}\vert\mathbf{w})}d\mathbf{w} \nonumber \\
&= \int q(\mathbf{w}\vert\theta)\left[\log{q(\mathbf{w}\vert\theta)}-\log{p(\mathbf{w})}-\log{p(\mathcal{D}\vert\mathbf{w})}  \right]d\mathbf{w}
\end{align}
$$

그렇다면 기존의 수식은 다음과 같이 변경될 수 있을 것이다.

$$
\theta^{*} = \text{argmin}_{\theta}\int q(\mathbf{w}\vert\theta)\left[\log{q(\mathbf{w}\vert\theta)}-\log{p(\mathbf{w})}-\log{p(\mathcal{D}\vert\mathbf{w})}  \right]d\mathbf{w}
$$

이 수식은 대괄호 안의 수식을 $$\mathbf{w}$$에 대한 conditional expectation 취한 것과 같다. 이는 $$\mathbf{w} \sim q(\mathbf{w}\vert\theta)$$ 에서 Sampling된 $$\mathbf{w}$$ 값에 대한 평균값과 동일하다. 따라서 대괄호 안의 값에 대해서만 최적화가 중요한 것으로 판단할 수 있어 minimize해야할 objective function은 다음과 같이 간소화시킬 수 있다.

$$ \mathcal{F}(\mathcal{D},\theta) \simeq \log{q(\mathbf{w}\vert\theta)} - \log{p(\mathbf{w})} - \log{p(\mathcal{D}\vert\mathbf{w})} \quad \text{where} \quad \mathbf{w} \sim q(\mathbf{w}\vert\theta) $$

Neural Network 학습은 이 objective function을 최소화시킬 수 있는 parameter $$\theta$$를 찾는 방향으로 이루어지며 그 결과로 $$\mathbf{w}$$를 Sampling 할 수 있다.

$$\mathbf{w}$$에 대한 Sampling이 이루어지기 때문에 동일 input $$\mathbf{x}$$에 대해서도 $$\mathbf{y}$$ 값이 다르게 출력될 수 있다. 따라서 여러번 Sampling 시행 후, $$\hat{\mathbf{y}}$$ 값은 반복 수행한 결과의 평균값으로 출력한다.

이러한 방법으로 인해 우리는 예측값의 Uncertainty를 구할 수 있으며, $$\hat{\mathbf{y}}$$ 에 대한 분산을 예측값에 대한 Uncertainty로 활용할 수 있다.






#### 참고문헌

1. [Bayesian Deep Learning](https://www.edwith.org/bayesiandeeplearning/joinLectures/14426)
2. [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)