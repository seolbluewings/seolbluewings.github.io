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
|:-:|:----|
|^| <span style="font-size:80%">^뒤에 있는 문자로 문자열이 시작되는 경우 추출</span>|
|\$| <span style="font-size:80%">\$ 앞에 있는 문자로 문자열이 끝나는 경우 추출</span>|
|.| <span style="font-size:80%">어떠한 문자가 들어가도 상관 없다(wild-card)</span>|
|\s| <span style="font-size:80%">공백(whitespace)을 의미</span>|
|\S| <span style="font-size:80%">공백을 허용하지 않음(non-whitespace)</span>|
|*| <span style="font-size:80%">* 앞에 있는 문자가 몇개가 존재하든(0개 포함) 추출</span>|
|?*| <span style="font-size:80%">* 앞에 있는 문자가 0번 또는 1번 존재하는 경우 추출</span>|
|+| <span style="font-size:80%">+ 앞에 있는 문자가 최소 1회 이상 반복되어야 추출</span>|
|[]| <span style="font-size:80%"> 대괄호([])안에 포함된 문자 중 하나와 매칭되는 경우 추출</span>|
|()| <span style="font-size:80%"> 괄호 안에 포함된 내용 추출</span>|
|[a-zA-Z]| <span style="font-size:80%"> 대괄호 안 - 사용하면 두 문자 사이의 범위에 해당하는 문자 추출</span>|
|[^]| <span style="font-size:80%"> 대괄호 안의 ^는 반대를 의미, 대괄호 안 문자를 제외한 경우 추출</span>|

먼저 Python에서 정규 표현식을 사용하고자 한다면, 정규식 라이브러리(re)를 호출해야 한다. re 라이브러리에는 re.search와 re.findall이 있는데 search는 찾고자 하는 정규표현식의 유무를 True/False로 Return해주는 반면, findall은 정규표현식을 만족하는 문자열을 리스트 형식으로 출력해준다.

위의 정규표현식을 이용해 패턴을 추출하는 방식은 다음과 같다.

- 패턴 추출하기 : [0-9+] 한자리 이상의 숫자 추출
```
import re
x = 'Suwon Bluewings won th K League in 2004 and 2008'
y = re.findall('[0-9]+',x)
print(y)
```

- Greedy한 탐색 및 Non-Greedy 탐색

정규표현식에서 Greedy하다는 것은 주어진 조건을 만족하는 경우 더 긴 문자열을 출력하려는 경향을 의미한다. Greedy한 문자열 탐색이 기본 옵션이고 Non-Greedy한 조건을 주고 싶다면, ?표기를 추가하여 가능한 짧은 문자열을 선택하도록 만든다. 

```
x = 'League : 1998,1999,2004,2008 / FA Cup : 2002,2009,2010,2016,2019 /'
y= re.findall('^L.+/',x)
print(y)
y= re.findall('(^L.+?)/',x)
print(y)
```

실질적으로 정규표현식은 다음과 같이 사용해볼 수 있다. 주어진 문장에서 이메일 주소만 추출하고자 한다면, 다음과 같이 입력할 수 있을 것이다.
```
lin = 'From seolbluewings@wooricard.com Wed May 18 08:55:34 2022'
y = re.findall('^From ([^ ]+@[^ ]+)',lin)
print(y)
```
아이디 생성 규칙까지 엄밀하게 적용한 정규표현식은 아니지만, 이 정규표현식은 다음의 절차를 거쳐서 문자열을 찾는다. 1. @를 찾으며 2. @를 찾았다면 앞뒤로 공백이 아닌 문자를 찾는다. 3. 해당 문자가 1개 이상 존재해야 하며 4. ()를 통해서 출력할 범위를 지정해준다.

To be continued...




#### 참조 문헌
1. [파이썬을 이용한 웹 스크래핑](https://www.boostcourse.org/cs201/joinLectures/179628) <br>

