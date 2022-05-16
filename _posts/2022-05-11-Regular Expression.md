---
layout: post
title:  "Regular Expression"
date: 2022-05-11
author: seolbluewings
categories: Data
---

(작성중...)

정규표현식은 프로그래밍 과정에서 문자열을 다룰 때, 문자열의 일정한 패턴을 표현하는 형식 언어이다. 본격적인 웹 스크래핑/크롤링에 앞서 스크래핑/크롤링 시 찾고자하는 문자를 더욱 쉽게 찾게 해줄 정규표현식을 정리해보고자 한다.

정규표현식에서 자주 사용하는 문자는 다음과 같다.

|문자|의미|
|:---:|:---|
|^| <span style="font-size:80%">^뒤에 있는 문자로 문자열이 시작되는 경우 추출</span>|
|\$| <span style="font-size:80%">\$ 앞에 있는 문자로 문자열이 끝나는 경우 추출</span>|
|.| <span style="font-size:80%">어떠한 문자가 들어가도 상관 없다(wild-card)</span>|
|\s| <span style="font-size:80%">공백(whitespace)을 의미</span>|
|\S| <span style="font-size:80%">공백을 허용하지 않음(non-whitespace)</span>|
|*| <span style="font-size:80%">* 앞에 있는 문자가 몇개가 존재하든(0개 포함) 추출</span>|
|?*| <span style="font-size:80%">* 앞에 있는 문자가 0번 또는 1번 존재하는 경우 추출</span>|
|+| <span style="font-size:80%">+ 앞에 있는 문자가 최소 1회 이상 반복되어야 추출</span>|
|[]| <span style="font-size:80%"> 대괄호([])안에 포함된 문자 중 하나와 매칭되는 경우 추출</span>|
|[a-zA-Z]| <span style="font-size:80%"> 대괄호 안에 -를 사용하면 두 문자 사이의 범위에 해당하는 문자 추출</span>|
|[^]| <span style="font-size:80%"> 대괄호 안에서의 ^는 반대를 의미함, 대괄호 안 문자를 제외한 경우 추출</span>|

먼저 Python에서 정규 표현식을 사용하고자 한다면, 정규식 라이브러리(re)를 호출해야 한다.



#### 참조 문헌
1. [파이썬을 이용한 웹 스크래핑](https://www.boostcourse.org/cs201/joinLectures/179628) <br>

