---
layout: post
title:  "Multi-Armed Bandit Problem"
date: 2022-11-19
author: seolbluewings
categories: Statistics
---

[작성중...]

Multi-Armed Bandit(이하 MAB) 문제는 A/B 테스트의 확장 형태라고 볼 수 있고 소비자에게 여러개의 제품 중 어떤 것을 노출시킬 것인가? 등의 문제를 해결하기 위한 알고리즘으로 활용된다.

![MAB](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/MAB.png?raw=true){:width="70%" height="30%"}{: .aligncenter}

MAB는 수익률이 각기 다른 여러개의 슬롯머신 중 수익을 최대화시키기 위해서 어떠한 슬롯머신의 손잡이를 당길 것인가? 라는 질문을 해결하기 위한 알고리즘으로 한정된 시간, 예산 조건 하에서 어떠한 슬롯 머신이 가장 최적의 결과를 가져다주는지를 탐색(exploration)하고 그 결과를 활용(exploitation)하면서 기대수익을 극대화하는 전략이라 할 수 있다.

앱에서 고객에게 N개의 상품 중 단 1개만을 노출 가능한 상황인데 어떤 것을 노출시키는 것이 고객의 클릭/구매를 많이 유도할 수 있는가? 라는 질문과 앞서 여러 슬롯머신 중 어떠한 슬롯머신을 당길 것인가? 라는 질문은 결이 같다. 따라서 공급자 입장에서 여러개의 선택지가 있지만, 고객에게는 제한적인 개수만 보여줄 수 있다면 MAB 알고리즘을 활용해 가장 기대보상이 큰 것을 노출시키면 된다.

MAB는 강화학습의 핵심 아이디어 중 하나인 탐색(Exploration)과 활용(Exploitation)을 활용하여 가장 큰 보상이 기대되는 슬롯을 찾는다. 따라서 탐색과 활용에 대한 정의가 MAB를 이해하기 위해 필요하다.

- 탐색(exploration) : 기대값이 가장 큰 선택지를 더욱 확실하게 찾기위해 선택하지 않을 것 같은 불확실성이 높은 선택지도 사용자에게 제시하는 행위
- 활용(exploitation) : 현재까지의 데이터를 바탕으로 사용자가 가장 많이 선택한 것과 동일한 것을 사용자에게 추천하는 행위

예를 들어 앱에서 고객에게 배너를 노출하는 상황을 가정하자. 각 고객의 과거 데이터를 바탕으로 고객이 클릭할 것으로 생각되는걸 노출시키는 것이 활용(exploitation)이다. 반대로 고객의 과거 데이터를 살펴보았을 때, 클릭할 것으로 생각되지는 않으나 관심이 있을지도 모르는 배너를 노출시켜보는 것이 탐색(exploration)이라 할 수 있다.

exploration이 너무 적을 경우, 무엇이 최적의 배너인지 정보가 부족한 상태에서 exploitation만 진행하게 된다. 따라서 최종적인 기대수익이 극대화될 수 있는지 장담할 수 없다. 반대로 exploration이 지나치게 많은 경우, 기회비용이 과도하게 발생하여 마찬가지로 기대수익이 극대화되지 못한다.

이처럼 exploration-exploitation 둘 사이에는 trade-off 관계가 존재하여 현재의 단기적인 이익을 극대화하는 exploitation과 더 많은 정보를 수집하려는 exploration의 비중을 적절히 조절할 필요가 있다.

#### Thompson's Sampling

MAB 문제를 풀기위해서 대표적으로 사용되는 알고리즘은 $$\epsilon$$-Greedy 방법, UCB(Upper Confidence Bound) 방법, Thompson's Sampling 3가지가 있다. 이 3개 중 이번 포스팅에서는 Thompson's Sampling에 대해 알아보고자 한다.

Thompson's Sampling이란 과거의 데이터를 이용하여 슬롯머신에서 제공하는 보상(reward)의 분포를 추정하고 그 분포를 통해서 가장 높은 보상을 줄 슬롯머신을 선택하는 알고리즘이다. MAB 문제를 풀기위해 활용되는 3가지 방법론 중 가장 성능이 좋은 것으로 알려져있으며 Bayesian 방법이므로 이번 포스팅에서 소개하고자 한다.

1. 슬롯머신이 K개가 있고 K개의 슬롯머신 arm에 대해 사전분포를 설정한다. 해당 사전분포로부터 sampling 진행하여 확률값이 가장 높게 나오는 arm을 선택한다.
2. arm을 선택한 결과를 반영하여 사후분포를 계산한다.
3. 2번 단계를 거쳐 생성된 Posterior를 새로운 사전 분포로 간주하고 일정시간/횟수만큼 1번/2번 단계를 반복한다.

이를 가장 대표적인 Thompson's Sampling 적용사례인 K개의 광고 배너 중 어떠한 배너가 가장 높은 클릭률(CTR)을 이끌어낼 것인가? 란 문제에 대입해보면 다음과 같이 문제를 해결해갈 수 있다.

총 K개의 배너가 있고 이중 k번째 배너는 $$\pi_{k}$$의 확률로 클릭이 발생한다고 가정한다. $$X_{k}$$를 k번째 배너를 클릭하는 사건에 대한 확률변수라 정의 한다면, 이 확률변수는 확률 $$\pi_{k}$$를 parameter로 갖는 베르누이 분포를 따른다고 $$X_{k} \sim \text{Ber}(\pi_{k})$$ 볼 수 있다.

또한 parameter $$\pi_{k}$$ 에 대한 사전분포로는 $$\pi_{k} \sim \Beta(\alpha_{k},\beta_{k})$$ 로 Beta 분포를 설정한다. 이 때, $$\alpha,\beta$$를 각각 1로 설정하면, uniform 분포와 동일해져 non-informative prior로 설정을 할 수가 있다.
















Bayesian A/B Test에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/python_study/blob/master/01.study/bayesian_AB_test.py)에서 확인 가능하다.


#### 참조 문헌
1. [Bayesian AB Test](https://assaeunji.github.io/bayesian/2020-03-02-abtest/) <br>
2. [Bayesian A/B Testing with Expected Loss](https://miistillery.me/bayesian-ab-testing/)
