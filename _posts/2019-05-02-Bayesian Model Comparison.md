---
layout: post
title:  "Bayesian Model Comparison"
date: 2019-05-22
author: YoungHwan Seol
categories: Bayesian
---

베이지안 관점에서의 모델 비교는 모델 선택에 있어서의 불확실성을 확률로 나타내는 것을 바탕으로 한다. L개의 모델 $$\{\mathcal{M}_{i}\}$$ ($$\mathit{i}=1,2,...,L$$)을 비교하기로 하자. 이 때 $$\{\mathcal{M}_{i}\}$$는 관측된 데이터 집합 $$\mathcal{D}$$에 대한 확률분포를 의미한다. 

L개의 모델들 중에서 어떤 모델로부터 데이터가 만들어졌는지 불확실하며 이 불확실성은 prior $$\mathit{p}(\mathcal{M_{i}})$$로 표현된다. 이 상황에서 prior는 각 다른 모델에 대한 우리의 선호도를 표현해준다. 

$$\mathit{p}(\mathcal{D}\|\mathcal{M_{i}})$$는 각기 서로 다른 모델들에 대한 데이터로 보여지는 선호도를 나타내는 모델 증거(model evidence)라고 할 수 있고 이는 marginal likelihood라고 불리기도 한다. 모델 증거의 비율 $$\mathit{p}(\mathcal{D}\|\mathcal{M_{i}})/\mathit{p}(\mathcal{D}\|\mathcal{M_{j}})$$는 베이즈 요인(Bayes factor)라고 불리기도 한다.

훈련 데이터집합 $$\mathcal{D}$$가 주어졌다면, 우리의 관심사 posterior는 다음과 같을 것이다.

$$\mathit{p}(\mathcal{M_{i}}|\mathcal{D}) \propto \mathit{p}(\mathcal{M_{i}})\mathit{p}(\mathcal{D}|\mathcal{M_{i}})$$

모델의 사후분포 $$\mathit{p}(\mathcal{M_{i}}\|\mathcal{D})$$를 알게 되면, 다음과 같은 예측분포를 구할 수 있다.

$$\mathit{p}(y|{\bf x},\mathcal{D})=\sum_{i=1}^{L} \mathit{p}(y|{\bf x},\mathcal{M_{i}},\mathcal{D})\mathit{p}(\mathcal{M_{i}}|\mathcal{D})$$

이는 각 개별 모델의 예측분포 $$\mathit{p}(y\|{\bf x},\mathcal{M_{i}},\mathcal{D})$$를 모델의 posterior $$\mathit{p}(\mathcal{M_{i}}\|\mathcal{D})$$로 가중 평균을 구하여 종합적인 predictive distribution을 구하는 것이다.

모델의 평균을 구하는 방법 중 가장 간단하게 근사하는 방법은 확률이 가장 높은 모델을 사용하여 그 1개의 모델만을 이용해 예측을 하는 것이며 이를 모델 선택(model selection)이라고 한다.

매개변수 $${\bf \beta}$$에 의해 결정되는 모델의 경우 모델 증거,$$\mathit{p}(\mathcal{D}\|\mathcal{M_{i}})$$는 다음과 같이 구할 수 있다.

$$\mathit{p}(\mathcal{D}|\mathcal{M_{i}}) = \int \mathit{p}(\mathcal{D}|{\bf \beta},\mathcal{M_{i}})\mathit{p}({\bf \beta}|\mathcal{M_{i}})d{\bf \beta}$$

이는 prior 분포로부터 랜덤하게 추출된 parameter를 가지는 모델로부터 데이터가 생성되었다고 볼 수 있고 이 모델 증거는 $${\bf \beta}$$에 대한 posterior distribution을 계산할 때, 베이즈 정리 분모 부분에 해당한다. 

$$ \mathit{p}({\bf \beta}|\mathcal{D},\mathcal{M_{i}})=\frac{\mathit{p}(\mathcal{D}|{\bf \beta},\mathcal{M_{i}})\mathit{p}({\bf \beta}|\mathcal{M_{i}})}{\mathit{p}(\mathcal{D}|\mathcal{M_{i}})}$$

위에서 본 적분식을 이용하여 

우선 단일 매개변수 $$\beta$$를 갖는 모델을 생각해보자. $$\beta$$의 posterior 분포는 다음과 같은 수식에 proportional 할 것이다.

$$ \mathit{p}({\bf \beta}|\mathcal{D},\mathcal{M_{i}}) \propto \mathit{p}(\mathcal{D}|{\bf \beta},\mathcal{M_{i}})\mathit{p}({\bf \beta}|\mathcal{M_{i}})$$

다음과 같이 2가지 가정을 하자.
1. posterior distribution이 $$\beta_{MAP}$$(가장 가능성이 높은 값)에서 뾰족하게 솟아 있고 그 폭이 $$\Delta\beta_{posterior}$$라 표현한다.
2. prior distribution의 폭은 $$\Delta\beta_{prior}$$이며 평평한 형태를 가지고 있어 $$\mathit{p}(\beta)=1/\Delta\beta_{prior}$$라 표현한다.

![Bayesian_Model_Comparison](C://seolbluewings.github.io/assets/images/Figure3.12.PNG)

위에서 언급한 2가지 가정은 위의 그림 파일로 설명될 수 있으며 이를 적용하면 다음과 같은 식을 얻을 수 있다.

$$\mathit{p}(\mathcal{D}|\mathcal{M_{i}}) = \int \mathit{p}(\mathcal{D}|\beta,\mathcal{M_{i}})\mathit{p}(\beta|\mathcal{M_{i}})d\beta \simeq \mathit{p}(\mathcal{D}|\beta_{MAP})\frac{\Delta\beta_{posterior}}{\Delta\beta_{prior}}$$

그리고 이후 과정에서는 편의상 $$\mathit{p}(\mathcal{D}\|\mathcal{M_{i}})$$를 $$\mathit{p}(\mathcal{D})$$로 표기하기로 한다. 여기에 로그를 취하면 다음과 같은 식을 얻을 수 있다.

$$ \log\mathit{p}(\mathcal{D}) \simeq \log\mathit{p}(\mathcal{D}|\beta_{MAP}) + \log(\frac{\Delta\beta_{posterior}}{\Delta\beta_{prior}})$$

이 식은 2가지 항으로 구성되어있으며 첫번째 항은 $$\beta_{MAP}$$를 바탕으로 데이터에 근사한 것으로 prior가 평평한 경우, log-likelihood에 해당한다. 두번째 항은 모델의 복잡도에 대한 penalty항이다. 일반적으로 $$\Delta\beta_{posterior}<\Delta\beta_{prior}$$ 이므로 $$\log{\frac{\Delta\beta_{posterior}}{\Delta\beta_{prior}}}$$ 의 값은 음수가 된다. $$\frac{\Delta\beta_{posterior}}{\Delta\beta_{prior}}$$의 값이 작아질수록(사후 확률이 고정된 특정 값으로 점점 수렴해 갈수록) log항의 절대값이 커지고 penalty가 커진다. 

$${\bf \beta}$$가 p차원이라면, 각각의 $$\beta$$에 대하여 비슷한 근사를 시행할 수 있다. 여기서 모든 $$\beta$$들이 같은 $$\frac{\Delta\beta_{posterior}}{\Delta\beta_{prior}}$$ 비율을 가졌다고 가정하자. 그러면 다음과 같은 식을 얻을 수 있다. 

$$ \log\mathit{p}(\mathcal{D}) \simeq \log\mathit{p}(\mathcal{D}|{\bf \beta_{MAP}}) + p\log(\frac{\Delta\beta_{posterior}}{\Delta\beta_{prior}})$$

가장 단순한 형태의 근사를 할 경우에도 penalty항의 크기는 p에 의해 선형적으로 증가한다. 대다수의 경우 모델의 복잡도가 증하감에 따라 첫번째 항의 크기는 증가하지만, 그에 따라 p의 크기가 증가할 것이므로 두번째 penalty항은 감소할 것이다. 최적 모델의 복잡도는 이 2개 항의 trade-off 관계에 의해 결정된다. 

1. 복잡한 모델을 사용하면 p값이 커진다.
2. 이 때 marginal likelihood의 확률값이 증가한다.
3. penalty term이 p에 의해 증가하므로 $$p(\mathcal{D})$$ 를 감소시킨다.
4. p의 수를 결정하는 것이 핵심적인 사항이 된다. 

![Bayesian_Model_Comparison](C://seolbluewings.github.io/assets/images/Figure3.13.PNG)

다음의 그림을 통해 marginal likelihood, $$p(\mathcal{D}\|\mathcal{M_{i}})$$가 어떻게 적당한 수준의 복잡도를 가진 모델을 선택하는지 알 수 있다. 

그래프에서 x축은 가능한 데이터 집합의 공간을 나타내며, 축 위의 각점은 특정 데이터 집합을 의미한다. 순차적으로 복잡도가 증가하는 3가지 모델$$\mathcal{M_{1}},\mathcal{M_{2}},\mathcal{M_{3}}$$ 이 있다고 하자. 

주어진 모델은 서로 다른 데이터 집합을 생성할 수 있다. 단순한 모델은 variance가 적으므로 단순한 모델을 기반으로 생성된 데이터 집합 간의 차이는 작을 것이며 위 그래프에서 파란색 영역으로 대응된다. 복잡한 모델의 경우는 더욱 다양한 종류의 서로 다른 데이터 집합을 생성해낼 수 있고 그 분포 $$p(\mathcal{D})$$는 더 넓은 공간에 존재하며 위 그래프에서 초록색 영역으로 대응된다. 

그래프에서처럼 특정 데이터 집합 $$\mathcal{D_{0}}$$의 경우 중간 정도의 복잡도를 가지는 모델 $$\mathcal{M_{2}}$$에서 가장 큰 모델 증거, $$p(\mathcal{D}\|\mathcal{M_{i}})$$를 가지는걸 확인할 수 있다. 

단순한 모델은 데이터에 잘 근사하지 못하며, 복잡한 모델은 예측 확률을 너무 넓은 데이터 집합에 퍼뜨리기 때문에 각각의 데이터 집합들에 대해 작은 확률을 할당한다.

베이지안 모델 비교 방법에서는 고려하고 있는 모델들의 집합 안에 실제 모델 분포가 존재한다고 가정한다.  이 때 베이지안 모델 비교법은 평균적으로 올바른 모델을 선택한다. 

두개의 모델 $$\mathcal{M_{1}},\mathcal{M_{2}}$$가 있다고 가정하자. 그리고 true model은 $$\mathcal{M_{1}}$$라고 하자.

두 모델 증거의 비율인 베이즈 요인(Bayes factor)을 실제 데이터 집합의 분포에 대해 평균을 구하면 그 기대값을 다음의 형태로 얻을 수 있다. 

$$ \int p(\mathcal{D}|\mathcal{M_{1}}) \log{\frac{p(\mathcal{D}|\mathcal{M_{1}})}{p(\mathcal{D}|\mathcal{M_{2}})}}d\mathcal{D} $$

이는 KL divergence의 형태이며, 두 분포가 일치해서 0값을 가질 때를 제외하면 항상 0보다 크다. 따라서 평균적으로 베이즈 요인은 올바른 모델을 선택하게 된다. 


