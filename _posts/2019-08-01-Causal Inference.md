---
layout: post
title:  "Causal Inference"
date: 2019-08-01
author: YoungHwan Seol
categories: Bayesian
---

먼저 심슨의 역설(Simpson's paradox)에 대해 언급하도록 하자. 심슨의 역설이란 전체 집단에서 나타나는 통계적 연관성(statistical association)이 하위 집단에서는 동일하게 유지되지 않는 현상을 의미한다.

다음과 같은 사례를 생각해보자. 새로운 약이 개발되었고 새로운 약을 먹은 집단과 그렇지 않은 집단에 대해 환자의 회복 수준을 비교를 한다고 하자. A/B 표기는 총 B명 중 A명이 회복되었다는 의미다. 

||새로운 약 복용|새로운 약 복용 X|
|------|---|---|
|남성|81/87 (93%)| 234/270 (87%)|
|여성|192/263 (73%)|55/80 (69%)|
|전체|273/350 (78%)|289/350 (83%)|

다음의 표를 보고 우리는 다음과 같은 해석을 할 수 있다.

- 남성 환자의 경우, 여성 환자의 경우를 나누어 보면 신약을 복용했을 때 더 높은 회복률을 보이기 때문에 환자의 성별을 알면 신약을 처방해야 한다고 볼 수 있다.
- 환자의 성별을 분리하지 말고 따져보자. 이 때는 신약을 복용하지 않았을 때 더 높은 회복률을 보인다. 따라서 환자의 성별을 알지 못할 경우, 신약을 처방해서는 안 된다.

두가지 결론은 굉장히 이상하다. 신약이 남성과 여성 모두에게 이롭다면 성별 가리지 않고 모두에게 이로워야 하는데 Total에 해당하는 부분으로만 해석하면 그렇지 않다는 결론을 내리게 된다.

이러한 경우 전체 데이터보다 그룹으로 나누어진 데이터가 더 구체적이며 타당한 결론을 내린다고 볼 수 있다. 그러나 그룹으로 나누어진 데이터가 반드시 정확한 답을 제공하는건 아니다.

이번에는 환자의 성별 대신 신약 복용 여부에 따른 혈압이 기록된 표를 살펴보도록 한다. 그리고 약을 복용하면, 환자들의 혈압을 낮춤으로서 회복에 영향을 미친다는 사실을 알고 있다고 하자.

||새로운 약 복용 X|새로운 약 복용|
|------|---|---|
|혈압 감소 |81/87 (93%)| 234/270 (87%)|
|고혈압 유지|192/263 (73%)|55/80 (69%)|
|전체 |273/350 (78%)|289/350 (83%)|

이번에는 하위 집단(혈압에 따라 구분된 집단)의 관점에서 보았을 때, 신약은 효과가 있다고 보기 어렵다. 그러나 전체 데이터를 보면 신약은 효과가 있다.

앞선 예시처럼 해당 실험의 목적은 질병 회복 가능성에 대한 치료의 전반적 효과를 알아보는 것이다. 그러나 혈압을 낮추는건 치료가 회복에 영향을 미치는 여러가지 단계들 중 하나의 단계이기 때문에 혈압 기준으로 결과를 나누는 것은 무의미하다. 따라서 전체 자료를 바탕으로 약의 복용을 권장해야 한다.

이렇게 치료가 혈압 감소에 영향을 미치고 혈압 감소가 질병 회복에 영향을 준다는 정보(사실)를 데이터로부터 얻을 수 없다.

데이터에 숨은 인과 관계에 대해 접근하기 위해 다음의 4가지가 필요하다.

1. 인과성(causation)에 대한 정의
2. 인과 가정(causal assumptions)에 대한 인과 모형(causal models)을 만드는 것
3. 인과 모형의 구조를 데이터의 특징과 연결시키는 것
4. 모형과 데이터에 포함된 인과 가정의 결합으로부터 결론을 내는 것

먼저 인과성(causation)에 대해 이야기하자.

변수 B가 변수 A에 의존하면 변수 A는 변수 B의 원인(cause)이다. B가 A에 따라 값을 결정하면, A는 B의 원인이 된다. (A is a cause of B if B listens to A and decides its value in response to what it hears.)

구조적 인과 모형(structural causal model)은 확률변수 집합(U,V)와 이에 대한 함수($$f$$)로 구성된다.

앞서 변수 B가 변수 A에 의존하면 변수 A는 변수 B의 원인이라 하였는데, 구조적 인과 모형의 구성 요소를 통해 이야기하자면, B의 값을 결정짓는 함수 $$f$$에 A가 변수로 들어갈 때 변수 A는 변수 B의 직접적 원인(direct cause)가 된다.

확률변수의 집합을 U와 V, 2개로 구분한 이유는 2가지 변수가 갖는 성격이 다르기 때문이다. U는 외생 변수(exogenous variable)이며 외생 변수는 어떤 이유로 해당 데이터가 발생했는지 설명하지 못한다. V는 내생 변수(endogenous variable)로 내생 변수는 적어도 하나의 외생 변수의 자손(child)이다.

다음과 같은 예시를 통해 설명할 수 있다.

![CI](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/VS1.PNG?raw=true){width="30%" height="30%}

a는 교육년수, b는 근무년수 c는 급여를 나타낸다. 이때, $$U=\{a,b\}$$, $$V=\{c\}$$ $$f_{c} : c=a + 2b$$ 라고 할 수 있다. a,b가 모두 $$f_{c}$$에 나타나므로 a,b는 모두 c의 direct cause라 할 수 있다.

이처럼 그래프 모형은 변수 U와 변수 V를 나타내는 노드와 함수 $$f$$를 표현하는 링크(화살표)로 구성된다. 

##### 개입(intervention)

인과관계를 알 수 있는 가장 좋은 방법은 무작위 통제(Randomized Controlled) 실험을 진행하는 것이다. 어떤 처치(treatment) 혹은 개입(intervention)이 효과가 있는지 확인하기 위해 우리는 먼저 treatment group과 controlled group으로 분리를 한다.

결과에 영향을 주는 단 1개의 변수를 제외한 나머지 모든 요인이 동등하다고 했을 때, 단 1개의 요인만을 변하게 함으로써 결과가 어떻게 바뀌는지 확인할 수 있고 변화가 있던 단 1개의 요인으로 인해 결과가 바뀐 것을 확인할 수 있다. 이 때 controlled group과 treatment group 집단의 평균적 동질성이 보장되어야 한다.

그러나 단 1개의 요인을 제외한 나머지 요인을 모두 통제하는 실험을 진행하는 것은 아주 어려운 일이다.대신 데이터를 기록하는 관측 연구(observational study)를 수행할 수 있는데 관측 연구의 경우 인과 관계를 단순한 상관 관계로부터 풀어내기가 어렵다.

다음의 관계를 살펴보자.

![CI](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/causal1.jpg?raw=true){width="30%" height="30%}

X는 아이스크림 판매량, Y는 범죄율, Z는 기온이다.






$$
\begin{align}
	p(C_{k}, \mathbf{x}) &= p(C_{k})p(x_{1},...,x_{k} \mid C_{k}) \\
	&= p(C_{k})p(x_{1} \mid C_{k})p(x_{2} \mid C_{k},x_{1})\cdot\cdot\cdot p(x_{n}\mid C_{k},x_{1},....,x_{n-1})
\end{align}
$$

Naive Bayes에서는 $$C_{k}$$가 주어진 경우, $$x_{i}$$와 $$x_{j}$$가 독립이라는 가정을 한다. 즉 조건부 독립에 대한 가정이 있는 셈이다. Naive Bayes에서 조건부 독립은 다음과 같이 표기할 수 있다.

$$
\begin{align}
	p(x_{i} \mid C_{k}, x_{j}) &= p(x_{i} \mid C_{k}) \\
	p(x_{i} \mid C_{k}, x_{j},x_{k}) &= p(x_{i} \mid C_{k})
\end{align}
$$

따라서 Naive Bayes 모델은 다음과 같이 표현될 수 있다.

$$
\begin{align}
	p(C_{k}\mid \mathbf{x}) &\propto p(C_{k},\mathbf{x}) \\
    &\propto p(C_{k})p(x_{1}\mid C_{k})p(x_{2}\mid C_{k}) \cdot\cdot\cdot p(x_{n}\mid C_{k}) \\
    &\propto p(C_{k})\prod_{i=1}^{n}p(x_{i}\mid C_{k})
\end{align}
$$

결국 Naive Bayes 모델은 가장 가능성 높은 class를 찾아내는 것으로 다음과 같이 $$\hat{y} = argmax_{k \in \{1,...,K\}} p(C_{k})\prod_{i=1}^{n}p(x_{i}\mid C_{k})$$ 로 표현할 수 있다.

#### Using Bayes Classifier as a Generative Model

이번에는 Bayes Classifier를 이용하여 새로운 데이터를 generate 하는 과정을 진행해보고자 한다.

다음의 notation을 따르기로 하자.

- $$x$$ : input data
- $$y$$ : data label
- $$p(x \mid y)$$ : label y가 given일 때, x의 확률
- $$p(x \mid y=1)$$ : label $$y=1$$일 때, x의 확률
- $$p(y \mid x)$$ : data x가 주어진 상황에서 y의 확률 (분류 문제의 목표이기도 하다)

생성 모델(generative model)에 관한 논의를 진행하고자 하는 과정에서 우리는 $$p(y \mid x)$$ 대신 $$p(x \mid y)$$를 학습하는 것에 주목한다. 우리의 목표는 $$y=y_{i}$$로 class가 주어진 상황에서 $$p(x \mid y=y_{i})$$의 값을 구하는 것이다.

문제를 간단하게 만들기 위해 지금부터 다루는 input data가 모두 가우시안 분포를 따를 것이라고 가정하며 우리는 y의 label이 주어진 상황에서 $$p(x \mid y=y_{i})$$가 어떤 가우시안 분포를 따르는지 알아내고자 한다.

$$p(x \mid y=y_{i})$$의 분포를 알아내기 위한 과정은 다음과 같다.

1. class $$y_{i}$$에 해당하는 모든 data point $$x_{i}$$를 찾아낸다.
2. 이 때의 $$x_{i}$$ 값들을 가지고 $$\mu_{y_{i}}$$와 $$\sigma_{y_{i}}^{2}$$를 계산한다.

데이터를 생성하는 과정은 [Kaggle](https://www.kaggle.com/c/digit-recognizer/data)에 업로드 되어진 MNIST 데이터를 활용할 것이다.

우선 필요한 라이브러리와 데이터를 불러온다.

~~~
import os
from builtins import range, input
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal as mvn
import matplotlib.pyplot as plt

def get_mnist():
    df = pd.read_csv(r'C:\Users\user\Python\train.csv')
    data = df.as_matrix()

    # data를 random하게 섞는다.
    np.random.shuffle(data)
    # 데이터 X와 label Y 2가지 파트로 구분한다.
    X = data[:, 1:] / 255.0 # pixels values are in [0, 255] => Normalize the data
    Y = data[:, 0]
    return X, Y
~~~

BayesClassifier란 class를 만든다. 이 class는 3가지 단계로 구성되어 있다. 첫째, fit함수는 모델을 데이터에 fit하는 것이다. 둘째, y의 class가 주어진 상황에서 sampling 한다. 셋째, y를 sampling한다.

~~~
class BayesClassifier:
    def fit(self, X, Y):
        # assume classes ∈ {0, ..., K-1}
        self.K = len(set(Y))

        self.gaussians = list()

        for k in range(self.K):
            Xk   = X[Y == k]         # class k일 때의 모든 X_{k}를 불러온다.
            mean = Xk.mean(axis=0)   # 불러온 X_{k}의 평균을 계산
            cov  = np.cov(Xk.T)      # 불러온 X_{k}의 분산을 계산

            self.gaussians.append({
                "m": mean,
                "c": cov
            })

    def sample_given_y(self, y):
        g = self.gaussians[y]
        return mvn(mean=g["m"], cov=g["c"], tol=1e-12)

    def sample(self):
        y = np.random.randint(self.K)
        return self.sample_given_y(y)
~~~

이제 데이터를 불러오고 각 y class에 대한 X의 평균과 분산을 계산할 수 있다. y가 given인 상황에서 random sample과 mean에 대한 image는 아래의 시행 결과와 같다.

~~~
X, Y = get_mnist()
clf = BayesClassifier()
clf.fit(X, Y)

for k in range(clf.K):
    # show one sample for each class and the mean image learned in the process

    sample = clf.sample_given_y(k).reshape(28, 28) # MNIST images are 28px * 28px
    mean   = clf.gaussians[k]["m"].reshape(28, 28)

    plt.subplot(1, 2, 1)
    plt.imshow(sample, cmap="gray", interpolation="none") # interpolation is added to prevent smoothing
    plt.title("Sample")

    plt.subplot(1, 2, 2)
    plt.imshow(mean, cmap="gray", interpolation="none")
    plt.title("Mean")

    plt.show()
~~~

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/image.png?raw=true){:width="40%" height="40%"}{: .center}

데이터를 생성하는 과정은 다음과 같으며 이를 통해 생성된 데이터는 아래의 그림과 같다.

~~~
# generate a random sample
samples = list()

col_number = 4
row_number = 5

img_size   = 2.0

fig_size = plt.rcParams["figure.figsize"] # Current size: [6.0, 4.0]

fig_size[0] = img_size * col_number # width
fig_size[1] = img_size * row_number # heigh

fig, axes = plt.subplots(row_number, col_number)
fig.subplots_adjust(hspace=0.1)

for _ in range(col_number*row_number):
    row = _ // col_number
    col = (_ - row*col_number)
    axes[row, col].imshow(clf.sample().reshape(28, 28), cmap="gray", interpolation="none")
    axes[row, col].axis('off')

plt.rcParams["figure.figsize"] = fig_size
~~~

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/image.png2?raw=true){:width="70%"}{: .center}