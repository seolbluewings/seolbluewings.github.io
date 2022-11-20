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

MAB는 강화학습의 핵심 아이디어 중 하나인 탐색(Exploration)과 활용(Exploitation)을 활용하여 최적의 







Bayesian A/B Test에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/python_study/blob/master/01.study/bayesian_AB_test.py)에서 확인 가능하다.


#### 참조 문헌
1. [Bayesian AB Test](https://assaeunji.github.io/bayesian/2020-03-02-abtest/) <br>
2. [Bayesian A/B Testing with Expected Loss](https://miistillery.me/bayesian-ab-testing/)
