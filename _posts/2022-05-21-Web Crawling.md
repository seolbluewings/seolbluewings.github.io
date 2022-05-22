---
layout: post
title:  "Web Crawling"
date: 2022-05-11
author: seolbluewings
categories: Data
---

(작성중...)

웹 크롤링은 웹 페이지의 정보를 가져오는 행위를 의미한다. Python에서는 selenium과 beautifulsoup 라이브러리를 이용하여 크롤링이 이루어진다.

이번 포스팅에서는 selenium의 webdriver를 사용하여 크롤링하는 것을 정리하고자 한다. webdriver를 통해서 인터넷을 통해 사이트 접속, 버튼 클릭, 글자 입력과 같은 사람이 수행할 일을 코드를 통해 제어할 수 있으며, 크롬 버전에 맞는 webdriver를 사용하여 작업을 수행해야 한다.

이렇게 webdriver를 통해 접속하고자 하는 url의 HTML 정보를 가져올 수 있는데 BeautifulSoup 라이브러리를 통해 url상의 HTML 데이터에 대한 필요 정보를 parsing할 수 있다.

```
from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome(r"C:\Users\seolbluewings\Desktop\sample\chromedriver.exe") #Chromedriver 설치경로
driver.get('http://www.shinsegae-lnb.com/product/wineView?id=2058')
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
soup
```

BeautifulSoup 라이브러리의 'html.parser' 옵션을 통해 webdriver가 가져온 html 변수에 있는 문자열 데이터를 HTML 형식에 맞게 변형시킬 수 있다.

이제는 HTML 데이터에 원하는 정보가 어디있는지 파악할 수 있어야한다. HTML은 크게 다음과 같은 규칙으로 이루어져있다.
1. 시작과 끝이 있고 <태크>로 시작하여 </태그>로 종료된다.
2. 태그의 시작과 끝 사이에 웹 페이지 화면에 표시되는 정보가 포함된다.
3. 태그 기호 내에 속성을 가질 수 있다.

따라서 HTML에서 찾으려는 정보가 포함된 부분의 태그 정보를 활용해 원하는 데이터를 가져올 수 있다. 원하는 데이터가 있는 부분을 특정지을 때까지 정보를 추가하여 필요한 부분을 정확하게 지정해줄 수 있다.

(to be continued...)


#### 참조 문헌
1. [파이썬을 이용한 웹 스크래핑](https://www.boostcourse.org/cs201/joinLectures/179628) <br>
2. 데이터분석 실무 with 파이썬
