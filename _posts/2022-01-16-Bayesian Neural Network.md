---
layout: post
title:  "Bayesian Neural Network"
date: 2022-01-16
author: seolbluewings
categories: Statistics
---

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




#### 참고문헌

1. [인공지능 및 기계학습심화](https://www.edwith.org/aiml-adv/joinLectures/14705)
2. [Density Estimation with Dirichlet Process Mixtures using PyMC3](https://austinrochford.com/posts/2016-02-25-density-estimation-dpm.html)