---
layout: post
title:  "Bayesian Network(1)"
date: 2019-07-06
author: YoungHwan Seol
categories: Bayesian
---

베이지안 네트워크(Bayesian Network)는 확률변수 간의 관계를 노드(node)와 링크(link) 혹은 엣지(edge)를 사용해 그래프 모델로 표현하는 것이다. 이후의 논의를 진행하기에 앞서 다음의 용어들에 대하여 정리하고 가도록 한다.

\begin{enumerate}
	\item 노드(node) : 확률 변수 1개를 1개의 노드로 표현
    \item 링크(link) : 엣지(edge)라고 불리기도 하며 이는 확률변수들 사이의 확률적 관계를 나타낸다. 화살표로 표시된다.
\end{enumerate}

베이지안 네트워크(Bayesian Network)는 다음의 특징을 갖는다.
\begin{enumerate}
	\item 베이지안 네트워크는 방향성 그래프 모델(directed graphical-model)이다.
    \item 방향성이란, link가 양방향이 아닌 한쪽 방향으로만 관계가 성립하는 것이다.
\end{enumerate}

방향성 그래프를 통해 확률변수 사이의 관계를 표현하는 것이 편리하며, 비방향성 그래프 모델(undirected graphical model)의 경우는 확률변수들 사이의 제약을 표현하는데 편리하다. 비방향성 그래프 모델로는 마르코프 랜덤 필드(Markov Random Field)가 있다.







