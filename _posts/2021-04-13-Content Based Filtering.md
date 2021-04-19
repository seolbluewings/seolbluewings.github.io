---
layout: post
title:  "Content Based Filtering"
date: 2021-04-13
author: seolbluewings
categories: ML
---

[작성중]

추천 시스템이란 추천 대상자(이하 사용자)가 관심을 가질만한 컨텐츠(영화, 상품, 음악 등...)를 개인별 맞춤 형태로 추천해주는 것을 의미한다.

기본적으로 추천 시스템을 만들기 위해서는 사용자의 취향을 파악해야 한다. 추천 시스템을 활용하는 기업 입장에서는 사용자에게 구매/이용 가능성이 높은 컨텐츠를 노출 시킴으로써 매출을 증대시킬 수 있다. 유투브 같은 컨텐츠 플랫폼은 사용자가 자기들의 서비스를 더 오랫동안 이용할 수 있게 유도할 수 있다. 추천 시스템을 통해서는 충성고객을 유치할 수 있고 기업은 해당 사용자에 대한 더 많은 정보를 확보할 수 있어 사용자에게 더욱 정교한 추천을 해줄 수 있다. 정상적으로 작동한다면, 추천 시스템은 기업 입장에서 선순환 구조를 만들어낸다.

추천 시스템에는 Content Based Filtering과 Collaborative Filtering 방식이 있고 이번 포스팅에서는 Content Based Filtering에 대해 이야기 해보고자 한다.

#### 컨텐츠 기반 필터링(Content Based Filtering)

Content Based Filtering 방법은 컨텐츠의 속성을 이용, 사용자가 관심있는 컨텐츠 속성을 분석하고 사용자에게 새로운 컨텐츠를 추천해주는 방식이다. Content Based Filtering의 작동 방식은 다음과 같다.

내가 [There Will Be Blood] 라는 영화에 평점 5점을 줬다고 가정하자. 이 작품의 속성을 간단하게 정의해보면, 1. 아카데미 작품상 후보 2. PTA 감독의 작품 정도로 표현할 수 있다. 따라서 Content Based Filtering 활용 시, 추천 알고리즘은 나에게 또 다른 아카데미 작품상 후보인 [포드v페라리] 영화를 추천하거나 PTA 감독의 또 다른 작품인 [마스터], [매그놀리아] 등을 추천해줄 것이다.


![CBF](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/CBF.png?raw=true){:width="70%" height="70%"}{: .aligncenter}

위 이미지는 Content Based Filtering의 원리를 명쾌하게 보여줌과 동시에 Collaborative Filtering의 차이까지도 보여주는 가장 대표적인 이미지다. Content Based Filtering에서는 다른 사용자의 정보가 필요하지 않다.

다만 Content Based Filtering은 컨텐츠를 설명할 수 있는 데이터를 확보할 수 있어야 한다. 영화라는 컨텐츠를 감독, 수상, 출연배우 등으로 속성 정의를 하듯이 다른 컨텐츠를 Content Based Filtering으로 추천하고자 한다면 컨텐츠에 대한 속성 정의를 할 수 있어야 한다. 



#### 참조 문헌
1. [위키백과 TF-IDF](https://ko.wikipedia.org/wiki/Tf-idf) <br>
2. [한국어 임베딩](https://ratsgo.github.io/natural%20language%20processing/2019/09/12/embedding/)
