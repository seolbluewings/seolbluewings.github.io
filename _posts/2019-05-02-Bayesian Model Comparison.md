---
layout: post
title:  "Bayesian Model Comparison"
date: 2019-05-22
author: YoungHwan Seol
categories: Bayesian
---

베이지안 관점에서의 모델 비교는 모델 선택에 있어서의 불확실성을 확률로 나타내는 것을 바탕으로 한다. L개의 모델 $$\{\mathcal{M}_{i}\}$$ ($$\mathit{i}=1,2,...,L$$)을 비교하기로 하자. 이 때 $$\{\mathcal{M}_{i}\}$$는 관측된 데이터 집합 $$\mathcal{D}$$에 대한 확률분포를 의미한다. 

L개의 모델들 중에서 어떤 모델로부터 데이터가 만들어졌는지 불확실하며 이 불확실성은 prior $$\mathit{p}($$\mathcal{D}$$)$$로 표현된다. 이 상황에서 prior는 각 다른 모델에 대한 우리의 선호도를 표현해준다. 

$$\mathit{p}(\mathcal{D}|\mathcal{M_{i}})$$는 각기 서로 다른 모델들에 대한 데이터로 보여지는 선호도를 나타내는 모델 증거(model evidence)라고 할 수 있고 모델 증거의 비율 $$\mathit{p}(\mathcal{D}|\mathcal{M_{i}})/\mathit{p}(\mathcal{D}|\mathcal{M_{j}})$$는 베이즈 요인(Bayes factor)라고 불리기도 한다.

훈련 데이터집합 $$\mathcal{D}$$가 주어졌다면, 우리의 관심사 posterior는 다음과 같을 것이다.

$$\mathit{p}(\mathcal{M_{i}}|\mathcal{D}) \propto \mathit{p}(\mathcal{M_{i}})\mathit{p}(\mathcal{D}|\mathcal{M_{i}})$$

모델의 사후분포 $$\mathit{p}(\mathcal{M_{i}}|\mathcal{D})$$를 알게 되면, 다음과 같은 예측분포를 구할 수 있다.

$$\mathit{p}(y|{\bf x},\mathcal{D})=\sum_{i=1}^{L} \mathit{p}(y|{\bf x},\mathcal{M_{i}},\mathcal{D})\mathit{p}(\mathcal{M_{i}}|\mathcal{D})$$

이는 각 개별 모델의 예측분포 $$\mathit{p}(y|{\bf x},\mathcal{M_{i}},\mathcal{D})$$를 모델의 posterior $$\mathit{p}(\mathcal{M_{i}}|\mathcal{D})$$로 가중 평균을 구하여 종합적인 predictive distribution을 구하는 것이다. 


