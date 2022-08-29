---
layout: post
title:  "Web Crawling"
date: 2022-05-11
author: seolbluewings
categories: Data
---

웹 크롤링은 웹 페이지의 정보를 가져오는 행위를 의미한다. Python에서는 selenium과 beautifulsoup 라이브러리를 이용하여 크롤링이 이루어진다.

이번 포스팅에서는 selenium의 webdriver를 사용하여 크롤링하는 것을 정리하고자 한다. webdriver를 통해서 인터넷을 통해 사이트 접속, 버튼 클릭, 글자 입력과 같은 사람이 수행할 일을 코드를 통해 제어할 수 있으며, 크롬 버전에 맞는 webdriver를 사용하여 작업을 수행해야 한다.

이렇게 webdriver를 통해 접속하고자 하는 url의 HTML 정보를 가져올 수 있는데 BeautifulSoup 라이브러리를 통해 url상의 HTML 데이터에 대한 필요 정보를 parsing할 수 있다.

신세계 L\&B에서 제공하는 와인에 대한 정보를 크롤링 할 때, 기본적인 페이지 url 접근은 아래와 같은 코드로 진행할 수 있다.

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

원하는 정보를 포함하는 태그를 찾기 위해서는 F12를 누른 상태로 여러 태그를 눌러보면서 원하는 정보가 포함된 구역을 좁혀가는 방식으로 찾을 수 있다. 이 부분은 노가다가 필요한 부분이다.

그러나 태그 정보만으로 정보를 충분히 찾는 것이 어려울 때가 절대 다수다. 이럴 때는 각 태그의 hierarchy 구조를 반영하여 '>' 기호를 활용, 원하는 데이터의 위치를 명확하게 지정해줘야 한다. 그리고 원하는 데이터만 쏙 빼내고 싶다면, 데이터의 index 번호까지 활용하여야 한다.

만약 신세계 L\&B의 와인 상품 페이지에서 전체 상품 개수 정보를 가져오고자 한다면 다음과 같이 '>'와 index 정보를 활용하여 데이터를 추출할 수 있다.

```
page_url = get_search_page_url(1)
page_url = str(page_url)
driver.get(page_url)
time.sleep(1)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
int(re.findall('[0-9]+',str(soup.select('div.cont > p.total > span')))[0])
```

신세계 L\&B의 와인 정보를 Crawling하는 코드는 다음의 [링크](https://github.com/seolbluewings/python_study/blob/master/01.study/web_crawling.py)에서 확인 가능하다.


#### 참조 문헌
1. [파이썬을 이용한 웹 스크래핑](https://www.boostcourse.org/cs201/joinLectures/179628) <br>
2. 데이터분석 실무 with 파이썬
