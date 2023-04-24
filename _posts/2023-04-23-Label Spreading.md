---
layout: post
title:  "Label Spreading"
date: 2023-04-23
author: seolbluewings
categories: Statistics
---

[작성중...]

Label Spreading은 일부 데이터의 label이 존재하지 않는 데이터에서 semi-supervised learning 을 수행하는 모델 중 하나로 label이 주어진 일부 데이터셋의 정보만을 가지고 그래프 이론을 바탕으로 데이터에 label을 부여하는 작업을 수행한다. label이 존재하는 데이터로부터 label에 대한 정보가 label이 존재하지 않는 데이터로 퍼져가는 모델이라 볼 수 있다.

![label](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/label_spreading.png?raw=true){:width="90%" height="80%"}{: .aligncenter}

Label Spreading 모델은 유유상종이라는 굉장히 직관적인 가정에서 출발한다. 나 자신을 데이터 분포상 하나의 node로 정의할 때, 나의 node 주변에 있는 node들의 label은 나와 같은 label일 것이라 추정할 수 있다.

이는 준지도 학습(semi-supervised learning)의 기본 가정인 consistency와도 일치하는 부분이다. Label Spreading은 가까운 node들은 같은 label을 가져야한다는 local consistency와 같은 cluster에 있는 node들 끼리는 같은 label을 가져야한다는 global consistency 모두 만족한다.


만약 node i가 그룹 B라는 label에 속하는데 node i의 주변에 위치한 node들이 그룹 A란 label을 가지고 있다면, node i의 그룹 정보를 B에서 A로 수정하는 것이 Label Spreading 모델이다.

Label Spreading은 네트워크 상에서 긴밀하게 연결된 노드들의 하위 집합을 도출하는 네트워크 분석 방식 중 하나인 커뮤니티 탐지(Community Detection) 방법론 중에서 별도 지표 계산이 필요하지 않아 속도가 빠르다는 장점도 있다.

Label Spreading 알고리즘은 다음의 프로세스를 통해 진행된다.

1. 모든 node는 각자의 label을 가진 상태에서 시작한다
2. node의 순서를 random하게 배치한 리스트를 작성하고
3. 리스트 순서대로 node를 선택한 후, 해당 node의 주변 node가 지닌 label 중에서 가장 빈도가 높은 label로 해당 node의 label을 변경함
4. 모든 node가 최대빈도 label을 유일하게 가질 때까지 2,3의 과정을 반복

node의 최대빈도 label이 여러개 존재한다는 것은 이 node가 여러 커뮤니티의 중간에 끼어있다는 것을 의미하고 최대빈도 label이 유일하다는 것은 근처 데이터로 퍼진 label이 안정적 경계 상태에 도달했다는 것을 의미한다고 볼 수 있다.

이 알고리즘을 반복할수록 node i의 근처 영역에서 입지가 약한 label이 소멸하고 node i 근처에서의 밀도가 높은 label 중심으로 label이 재설정 된다.



(계속...)




#### 참조 문헌
1. [Label Propagation Algorithm](https://pizzathief.oopy.io/label-propagation-algorithm) <br>
2. [[네트워크이론] Label propagation algorithm for community detection](https://mons1220.tistory.com/168)

