---
title: '알고리즘 챕터 3장 : Decompositions of graphs'
date: 2020-03-09 22:14:00
categories: [Study Note, Algorithms]
tags: [Algorithms, DFS]
use_math: true
seo:
  date_modified: 2020-03-09 03:21:32 +0900
---



해당 글은 [Algorithms. by *S. Dasgupta, C.H. Papadimitriou, and U.V. Vazirani*](http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf) 를 정리한 스터디 노트입니다.



## 3.1 Why graphs?

- 다양한 문제들을 그래프 형태로 간결하고 명확하게 표현할 수 있음
  - Graph coloring 으로 아래 문제들을 모델링 가능  
    $\approx$ 행정구역도를 색칠하는 문제  
    $\approx$ 각 수업의 시험시간을 겹치지 않게 하는 문제
- 그래프의 표기법(Notation)
  - $G=(V,E)$
  - 그래프는 크게 유방향 그래프(directed graph)와 무방향 그래프(undirected graph)로 나눌 수 있음
    - undirected graph의 경우, edge를 symmetric relation로 볼 수 있음 $e=\\{ x+y \\}$
    - directed graph의 경우, directed edges를 $e=(x,y)$ 로 표현할 수 있음

- 그래프의 표현(Representation)
  - 인접 행렬(adjacency matrix)
    - undirected graph → symmetric matrix
    - good) 특정 edge의 존재 여부를 constant time ($n=\lvert{V}\rvert$)으로 체크 가능
    - bad) 인접 행렬을 저장할 때 $O(n^2)$의 공간을 차지함, sparse할 경우 너무 낭비임
  - 인접 리스트(adjacency list)
    - $\lvert{V}\rvert$개의 linked list로 이루어짐
    - 각 vertex에 연결된 vertex들만 저장함
    - undirected graph인 경우 $2*\lvert{E}\rvert$ , directed graph인 경우 $\lvert{E}\rvert$
    - 즉, 총 자료 구조의 크기는 $O(\lvert{E}\rvert)$
    - 특정 edge 확인할 때는 더 이상 constant time이 아님
    - 하지만, iteratate하기 쉬움
    <!--  -->
- 인접 행렬, 인접 리스트? 어떤 표현이 좋을까?
  - 결국 주어진 vertex가 어떻게 연결되어 있는지가 중요함
  - 이는 edge connectivity가 중요하다는 뜻
  - G가 connected graph with a cycle인 경우, 
    $(sparse\rightarrow )\; \lvert{V}\rvert \leq \lvert{E}\rvert \leq \lvert{V}\rvert^2 \; (\leftarrow dense)$ 
  - 결국, $\lvert{E}\rvert$ 에 따라 선택하는 것이 중요.

<br/>

## 3.2  undirected graph의 깊이우선탐색

### 3.2.1 Exploring mazes

- 그래프 탐색
  - Task : 주어진 vertex로부터 도달가능한 vertex들을 모두 찾아보자.
  - (미로탐색) 실뭉치와 분필
    - 이미 방문한 곳을 분필로 표시 (반복하는 것을 방지)
    - 실뭉치를 이용해, 원래 시작했던 장소로 돌아오게끔 해줌
  - (컴퓨터)
    - 각 vertex마다 boolean 변수로 표시
    - **stack** 구조의 2가지 연산을 통해 실뭉치 역할을 함
      - 새로운 곳을 갈 때는 $unwind$
      - 이전 곳으로 돌아올 때는 $rewind$
    - stack 구조를 explicitly 이용하기보다, recursion을 통해 implictly 알고리즘 구현

<br/>

특정 노드로부터 도달 가능한 모든 노드들을 찾는 알고리즘

```python
procedure explore(G,v)  
Input : G=(V,E) is a graph; v ∈ V  
Output : visited(u) is a set to true for all nodes u reachable from v  

visited(v) = true  
previsit(v)  
for each edge (v,u) ∈ E:  
  if not visited(u):	explore(u)  
postvisit(v)
```

- previsit, postvisit은 나중에 볼 연산 (선택사항)
  - previsit : 처음 발견한 해당 vertex에 대해 연산 수행
  - postvisit : 마지막으로 떠날 때 해당 vertex에 대해 연산 수행
  
- 위에서 정의한 탐색 알고리즘 과정이 제대로 돌아갈까?
  - (pf) 연결된 vertex 중에서 찾지 못한 vertex $u$ 가 있다고 가정  
    $v$ 에서 $u$ 로 가는 path $P$ 를 하나 선택  
    탐색알고리즘에 대해 마지막으로 탐색된 vertex를 $z$ 로 생각  
    Path $P$ 상에서 $u$ 로 가는 방향의 $z$ 이웃 노드인 $w$ 가 있다고 생각  
    즉, $z$ 는 탐색했지만, $w$ 는 탐색안함
    (모순) $z$ 에서 탐색과정을 통해 $w$ 로 향했기 때문
  - 위와 같은 증명 과정을 그래프론에서 자주함.  
    a streamlined induction.


​        

- 트리(Tree)
  - **A connected graph with no cycles**
  - Tree edges
  - back edges (이전방문했던 노드로 다시 돌아갈 때 표시하는 가상의 edge)
- 포레스트(Forest)
  - disjoint collection of trees

<br/>

### 3.2.2 Depth-frist search

깊이우선탐색(DFS, Depth-first search)

```python
procedure dfs(G)

for all v ∈ V:
  visited(v) = false

for all v ∈ V:
  if not visited(v): explore(v)
```

  - DFS 알고리즘 실행시간 분석
    - 1) 고정된 양의 작업(visited라고 표시함, 혹은 pre/postvisit)
      - 각 노드마다 마킹작업 수행
      - 총 연산량 $O(\lvert{V}\rvert$)
    - 2) 안가본 곳으로 향하기 위해, 인접 엣지들을 탐색하는 루프(loop)
      - $e=\\{x,y\\}$ 이므로 각 노드마다 엣지를 탐색
      - 엣지마다 2번씩 탐색되어짐
      - 총 연산량 $O(\lvert{E}\rvert$)
    - 따라서, DFS는 $O(\lvert{V}\rvert + \lvert{E}\rvert$) :star: 
      - 그래프의 인풋값인 $V$ 와 $E$ 에 대한 linear time임

<br/>

### 3.2.3 Connectivity in undirected graphs

- connected graph
  - 어떤 undirected graph가 연결되어 있다는 것은, 어떤 노드들을 잡아도 path가 존재한다는 것

- connected components
  - 각각의 component가 subgraph임
    - subgraph란, internally connected & no edges to the remaining nodes

- 특정 노드에 대한 탐색알고리즘은 그 노드가 포함된 connected component 하나를 알아내는 것임  
  → DFS를 통해, 그래프가 연결되어있는지 체크할 수 있음  
  → 하나의 노드에 숫자를 부여해서 포함된 connected component를 식별 가능

```python
procedure previsit(v)
ccnum[v] = cc
```

- cc : 0으로 초기화, DFS 과정에서 explore가 호출될 때마다 +1

- previsit / postvisit
  - 결국, DFS는 undirected graph의 연결구조를 linear time 내에 찾는 방법이라고 할 수 있음
  - 해당 노드를 첫번째로 발견할 경우 previsit
  - 마지막으로 떠날 경우 postvisit

```python
procedure previsit(v)
pre[v] = clock
clock = clock + 1 

procedrue postvisit(v)
post[v] = clock
clock = clock + 1
```

  - **Property** 어떤 노드 $u$ 와 $v$ 에 대해, 각각의 구간 $[pre(u),post(u)]$ 와 $[pre(u),post(u)]$ 는 서로 distjoint 이거나 포함관계임  
    WHY? stack 구조는 last-in, first-out 으로 동작하므로, 노드 $u$ 가 스택에 있는 동안의 시간흐름을 구간으로 생각가능

<br/>

## 3.3 directed graph의 깊이우선탐색

- 용어정리 (directed graph에서의 tree구조라고 가정) - 그림으로 이해하는 것이 재빠름!
  - 노드 관점의 연결관계
    - 탐색트리의 <u>root</u> ; 제일 상위인 노드를 말함
    - 노드의 <u>descendant</u> ; 그 노드보다 하위일 때
    - 한 directed edge에 대한 양쪽 노드가 서로  <u>parent</u>, <u>child</u> 관계임 (1촌수만 본다는 뜻)
  - 엣지 관점의 연결관계
    - tree edges : DFS 포레스트의 그냥 실제 solid한 실제 엣지들을 의미
    - 아래는 가상의 엣지라고 볼 수 있음
      - forward edges : 상위 노드가 nonchild descendant 노드로 가는 엣지
      - back edges : 하위 노드가 ancestor 노드로 가는 엣지
      - cross edges : neither descendant nor ancestor 노드 관계에서 가질 수 있는 엣지
- pre/post in directed graph
  - edge $(u, v)$ 에 대한 타입을 노드의 pre/post 구간값을 통해 알 수 있음
    - [u  [v    v]  u] : 우리가 알던 구조 ($u \rightarrow v$)
      - Tree/forward edges
    - [v  [u    u]  v] : $v \rightarrow u$ 
      - Back edges
    - [u  u]    [v   v] : 상관이 없다는 것임
      - Cross edges
  - 예시) u가 상위, v가 하위 단계로 연결되어 있으면, pre(u) < pre(v) < post(v) < post(u) 라고 볼 수 있음
- Directed acyclic graphs
  - cyclic / acyclic
    - 그래프에 cycle이 있다는 것은 circular path가 존재한다는 것으로 정의
    - cyclic이 아닌 그래프를 acyclic 이라고 함
  - **Property** driected graph가 cycle이 있다는 것은 그 그래프의 DFS가 back edge를 찾아내는 것과 동일
    - **proof)**  
      (<=)  (u, v)라는 back edge가 있다고 하자. 그러면, v에서 u로 가는 path와 함께 cycle 구성가능
      (=>) cycle이 있다고 가정하자. 따라서 DFS에 의해 가장 작은 pre number에 대해, 첫번째로 발견되는 노드가 있다. cycle에 있는 다른 노드들은 모두 그 첫번째 발견된 노드의 descendants다. 특히, 직전 노드는 ancestor로 향하는 노드이기 때문에, back edge의 정의에 해당함
  - dags, directed acyclic graphs
    - 의존성, 계층 관계를 표현하기 좋은 구조임
    - "어떤 유효한 순서로 작업을 해야할까?"와 같은 문제를 모델링함
      - 각 노드가 하나의 작업이고, 엣지 u->v 는 v의 선행으로 u가 작업되어야 한다는 것으로 표현가능
    - (만약 cycle 구조라면, 순서는 의미가 없다.)
    - 만약 dag 구조라면, linearize(or topologically sort)를 통해 순서를 표현할 수 있음
    - 어떤 형태의 dag 구조가 linearize 가능할까? 모두 가능함
    - **DFS를 통해 linearize한 순서를 찾을 수 있음**
      - <u>DFS를 통해 나온 post numbers를 내림차순으로 vertex를 나열하면 끝</u>
      - 즉, 마지막 훑은 애가 가장 상위노드라는 의미임
  - **property**  dag에서, 모든 edge는 작은 post number를 가진 노드로 향한다.
    - dag형태에서는 back edges형태를 가질 수 없음 (post number가 흐르는 방향을 생각)
    - 이 성질로 인해, dag의 노드들을 순서화하는데  linear-time algorithm으로 생각할 수 있음
    - <u>**acyclicity, linearizability, the absence of back edges during a DFS**</u> 라는 dag의 세가지 속성을 말해주는 성질임
  - Sink / source
    - highest post number 노드가 source
    - smallest post number 노드가 sink
  - **property** 모든 dag는 적어도 하나 이상의 source와 하나 이상의 sink를 가짐
    - 즉, 입출노드가 하나 이상은 있다.
    - source의 존재를 통해, linearization을 다른 방식으로 접근함 (linear time operation 증명필요)
      - 1) source를 찾고, 출력하고, 그래프에서 삭제함
      - 2) 그래프가 없어질 때까지 반복
      - 이게 모든 dag에 대해 성립할 수 있는 이유?  
        위의 성질에 따라 모든 dag는 하나 이상의 source가 존재함.



<br/>

## 3.4 Strongly connected components

### 3.4.1 Defining connectivity for directed graphs

- 연결성
  - undirected graph의 경우, 각각의 connected components에 대해 DFS를 실행시키면 끝
  - directed graph의 경우, connectivity를 다음과 같이 정의함
    - 두 노드 $u,v$ 가 서로 $connected$ 라는 것은, $u$ 에서 $v$ 로 가는 path가 존재 + $v$ 에서 $u$ 로 가는 path 존재
  - 이 연결 관계를 통해, $stongly \; connected \; components$ 를 정의할 수 있음
    - 즉, 이들은 directed graph를 $V$ 로 분할(parititions)하는 disjoint sets
- Strongly connected components
  - 각각의 strongly connected component를 하나의 메타노드로 표현해 메타그래프로 만들 수 있음
  - 이렇게 만들어진 메타그래프는, dag 형태가 됨
    - pf) 만약에 여러 개의 strongly connected components가 하나의 cycle을 포함한다면, 하나의 strongly connected components로 합쳐짐
  - 이를 정리하면  
    **(Property)** 모든 directed graph는 해당 그래프의 strongly connected components로 이루어진 하나의 dag다.
- 즉, directed graph는 2가지의 연결성 구조를 갖고 있음
  - 상위레벨에 dag로 간단하게 표현 (선형으로 표현 가능)
  - 하위레벨에 각 dag 속 세부 그래프로 표현

<br/>

### 3.4.2 An efficient algorithm

- directed graph를 strongly connected components로 분해하는 것은 굉장히 유용함  
  - 분해 과정은 linear time 으로 찾을 수 있음 (DFS+$\alpha$ 통해)
  

- **(Property 1)** 만약 $\text{explore}$ 서브루틴이 노드 $u$ 에서 시작하면, 그 서브루틴은 $u$ 로부터 도달가능한 모든 노드들을 방문했을 때 정확히 종료된다.
  - 즉, (메타 레벨에서) $sink$ strongly connected component인 노드에 대해 탐색 서브루틴을 호출하는 경우, 정확히 해당 component를 검색함 (= 메타레벨에서 종료한다는 뜻, 해당 component에서만 검색한다는 뜻)
  - 이를 통해 하나의 strongly connected component를 찾는 방법을 알 수 있지만, 여전히 2가지 문제 존재
    - **(A) 확실히 sink strongly connected component에 놓여져 있는 노드를 어떻게 찾을까?**
    - **(B) sink component를 찾았다면, 이후에 어떻게 반복해야할까?**
  - (A)를 확실히 찾을 수는 없지만, 반대로 $source$ strongly connected component를 찾는 방법은 존재! (↓)
- **(Property 2)** DFS를 통해 가장 높은 $\text{post}$ 숫자를 가진 노드는 $source$ strongly connected component에 반드시 놓여져 있다.
  - 이를 일반적으로 나타내면 아래와 같다. (↓)

- **(Property 3)** 만약 strongly connected components $C$ 와 $C'$ 에 대해서  $C$ 안에 있는 노드에서 $C'$안에 있는 노드로 가는 엣지가 존재할 때, $C$ 에서 가장 큰 $\text{post}$ 숫자는 $C'$ 에서 가장 높은 $\text{post}$ 숫자보다 크다.
  - pf) 2가지 경우 생각
    - 1번 case
      - DFS가 $C'$ 이전에 $C$ 를 탐색한다면, property 1에 의해서 $C$ 과 $C'$ 모두 끊기지 않고 탐색한 것
      - 따라서 $C$ 에 첫번째로 방문한 노드가 어떤 $C'$의 노드보다도 늦게 끝나므로 $\text{post}$ 숫자가 클 수밖에 없음
    - 2번 case
      - DFS에 의해 $C'$ 이 먼저 탐색된 경우, property 1에 의해서 $C$ 의 노드들을 보기도 전에 종료

- property를 다르게 해석해보면,
  - property 3를 "strongly connected components는 그들 내부의 가장 높은 $\text{post}$ 숫자들의 내림차순으로 정렬해서 linearized할 수 있다"라는 뜻으로 볼 수 있음
  - property 2는 그래프 $G$ 의 source strongly connected components 안에 있는 노드 하나를 찾을 수 있게 해줌
- 사실, 우리가 찾고 싶었던 노드는 source가 아니라 sink임. 어떻게 해결할 수 있을까? (A)
  - $G^{R}$ 라는 $reverse$ 그래프를 생각
  - $G^{R}$ 과 $G$ 는 같은 strongly connected components를 갖고 있음 (why?)
  - 따라서, $G^{R}$ 에 대해 DFS를 돌려서 나오는 $G^{R}$ 의 source components로부터 가장 높은 $\text{post}$ 숫자를 가진 노드가 나올 것
  - 결국 $G$ 의 관점에서 sink components로 볼 수 있음
  - $\therefore$ (A)를 $G^{R}$ 의 source를 찾는 것으로 해결함!
- 그렇다면, first sink component를 찾은 이후 어떻게 반복해야할까? (B)
  - Property 3을 이용
  - 첫번째 strongly connected component를 찾고 나서 그래프로부터 그 component를 지우고 나면, 나머지 노드들 중에서 가장 높은 $\text{post}$ 숫자를 가진 노드는 다시 sink component에 속함
  - 따라서 $G^{R}$ 에 대해 DFS를 처음 수행할 때, $\text{post}$ 숫자를 매기는 것을 기억시켜놨다가 순서대로 strongly connected components를 출력

<br/>

이를 알고리즘으로 정리하면 다음과 같다.

1. $G^{R}$ 에 대해 DFS 수행
2. $G$ 에 대해 undirected connected components 알고리즘(Section 3.2.3)을 수행  
   DFS를 하는 동안, step1로부터 나온 $\text{post}$ 숫자를 내림차순으로 노드들을 진행

알고리즘은 linear-time 임

- 구체적으로, linear term에 있는 상수만 기존 straight DFS의 2배임
- (Question) How does one construct an adjacency list represen- tation of GR in linear time? And how, in linear time, does one order the vertices of G by decreasing post values?)

<br/>

예시) Figure 3.9

- 첫번째 단계) $G^{R}$ 의 DFS를 탐색
  - G, I, J, L, K, H, D, C, F, B, E, A 로 나열
- 두번째 단계) $G^{R}$ 의 DFS 결과로 나온 $\text{post}$ 숫자 내림차순으로 recursive하게 components를 체크
  - {G, H, I, J, K, L}, {D}, {C, F }, {B, E}, {A} 를 찾아냄
    - G ($G^{R}$ 제일 높은 숫자) 를 기준으로 $G$ 에 대해 DFS 돌리고 제거한 뒤  
      다시 $D$ ($G^{R}$ 나머지에서 제일 높은 숫자)를 기준으로 $G$ 에 대해 DFS 돌리고...

<br/>

Crawling fast (읽기 자료)

- 우리가 배운 상황과 실제 상황(World Wide Web)은 다름
  - graph의 노드들이 아직 search되지 않은 경우도 있음
  - 따라서, recursion이라는 것은 의미가 없음
- 사실 여전히, Web을 탐색하는 알고리즘은 DFS와 굉장히 비슷함
  - 발견됐지만, 아직 탐색되지 않은 상태인 모든 노드들을 포함한 explicit stack로 유지
  - 사실 이 스택구조는 정확히 last-in, first-out list 구조가 아님
  - 

















