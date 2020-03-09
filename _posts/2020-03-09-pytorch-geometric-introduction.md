---
title: 예제를 통해 알아보는 PyTorch Geometric 기본개념c
date: 2020-03-09 12:06:00
categories: [Deep Learning, Tutorial]
tags: [Deep Learning, Graph Neural Network, PyTorch]
use_math: true
seo:
  date_modified: 2020-03-09 03:25:56 +0900
---



다음 글은 [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) 라이브러리 설명서에 있는  [Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#) 를 참고하여 작성했습니다.

<br/>

<img src="https://pytorch-geometric.readthedocs.io/en/latest/_static/pyg_logo_text.svg" width="400" height="400">

<br/>

최근 Graph Neural Network에 대해 관심이 많아 공부하던 도중  <kbd>PyTorch Geometric</kbd>라는 라이브러리를 알게되었습니다. 실제 코드를 작성해보지 않으면, 평생 사용할 수 없을 것 같아 해당 라이브러리의 Docs를 번역하여 글을 작성했습니다. 여러 예제를 통해 **PyTorch Geometric 라이브러리에서 등장하는 5가지 기본 개념**에 대해 살펴보겠습니다.

<br/>

- TOC 
{:toc}
<br/>

---

<br/>



## **그래프의 데이터 핸들링**

그래프는 단순히 노드(node 또는 vertex)와 그 노드를 연결하는 간선(edge)을 하나로 모아 놓은 자료 구조 입니다. 그래프를 구성하는 노드와 엣지들을 모아놓은 집합을 각각 $V, E$ 라고 했을 때, 그래프는 $G=(V,E)$ 로 표현할 수 있습니다.

PyTorch Geometric 에서 하나의 그래프는 `torch_geometric.data.Data` 라는 인스턴스로 표현됩니다.  
특히, 이 인스턴스는 다음과 같은 default 속성을 갖고 있습니다.

<br/>

- `data.x` : 노드특징 행렬
  - [num_nodes, num_node_features]
- `data.edge_index` : 그래프의 연결성
  - [2, num_edges]
- `data.edge_attr` : 엣지특징 행렬
  - [num_edges, num_edge_features]
- `data.y` : 학습하고 싶은 대상 (타겟)
  - 그래프 레벨 → [num_nodes, *]
  - 노드 레벨 → [1, *]
- `data.pos` : 노드위치 행렬
  - [num_nodes, num_dimensions]

<br/>

사실 위에 있는 속성들은 필수가 아니라 **옵션**입니다. 즉, 자신이 구성하고 싶은 속성들을 다양하게 모델링할 수 있습니다. 하지만 일반적으로 그래프 데이터는 노드와 엣지로 표현하기 때문에 위의 속성들로 표현하는 것이 적합해 보입니다.

기존의 `torchvision`은 이미지와 타겟으로 구성된 튜플 형태로 데이터를 정의했습니다.  
그와 다르게,  `PyTorch Geometric`은 조금 더 그래프에 직관적인 형태의 데이터 구조를 갖고 있습니다.

<br/>

그럼 Data 클래스를 사용해 그래프 인스턴스를 만들어보겠습니다.

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor(([-1], [0], [1], dtype=torch.float))

data = Data(x=x, edge_index=edge_index)
>>> Data(edge_index=[2, 4], x=[3, 1])
```
- `edge_index` : (2,4) 크기의 행렬 → 4개의 엣지조합
- `x` : (3,1) 크기의 행렬 → 3개의 노드와 각 노드의 특징은 1개

<br/>

일반적으로 엣지는 노드의 순서쌍으로 나타내는 경우가 많습니다.  
이런 경우, `contiguous()` 를 사용해 동일한 그래프로 표현할 수 있습니다. 

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                        [1, 0],
                        [1, 2],
                        [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
>>> Data(edge_index=[2, 4], x=[3, 1])
```

<br/>

<figure>
  <img src="https://pytorch-geometric.readthedocs.io/en/latest/_images/graph.svg" width="400" height="400">
    <figcaption><center>우리가 정의한 data 인스턴스의 실제 그래프</center></figcaption>
</figure>



<br/>

<br/>

<br/>

추가적으로, `torch_geometric.data.Data` 클래스는 다음과 같은 함수를 제공합니다.

- `data.keys` : 해당 속성 이름
- `data.num_nodes` : 노드 총 개수
- `data.num_edges` : 엣지 총 개수
- `data.contains_isolated_nodes()` : 고립 노드 여부 확인
- `data.contains_self_loops()` : 셀프 루프 포함 여부 확인
- `data.is_directed()` : 그래프의 방향성 여부 확인

<br/>

그래프론에서 자주 사용하는 루프, 고립된 노드, 방향성 등 그래프 특징을 반영한 함수들이 있네요.  
소스코드를 뜯어보면, 어떤 알고리즘을 사용했는지까지 알 수 있다는 생각이 듭니다.

<br/>

<br/>

<br/>

## **공통 벤치마크 데이터셋**

PyTorch Geometric은 **다양한 공통 벤치마크 데이터셋**을 포함하고 있습니다.  
해당 데이터셋의 종류는 [torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) 에서 확인할 수 있습니다.

각 데이터셋마다 그래프 데이터의 속성이 다르기 때문에 사용되는 함수가 다를 수 있습니다.  
그래프하면 빠질 수 없는 데이터셋인, **ENZYMES** 과 **Cora** 에 대한 예시를 살펴보겠습니다.

<br/>

다음은 **ENZYMES** 데이터셋을 불러오는 예제입니다.

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
>>> ENZYMES(600)

len(dataset)
>>> 600

dataset.num_classes
>>> 6

dataset.num_node_features
>>> 3
```

- `num_classes` : 그래프의 클래스 수
- `num_node_features` : 노드의 특징 수

<br/>

ENZYMES 데이터셋에는 6종류의 클래스를 가진 600개의 그래프가 있는 것을 확인할 수 있습니다.  
그래프 하나를 가져올 때는 어떻게 할까요?

```python
data = dataset[0]
>>> Data(edge_index=[2, 168], x=[37, 3], y=[1])

data.is_undirected()
>>> True

train_dataset = dataset[:540]
>>> ENZYMES(540)

test_dataset = dataset[540:]
>>> ENZYMES(60)

dataset = dataset.shuffle()
>>> ENZYMES(600)
```

- `슬라이싱`을 통해 하나의 그래프 데이터를 가져올 수 있습니다.
- `edge_index=[2, 168]` → 총 84개의 엣지
- `x=[37, 3]` → 총 37개의 노드와 3개의 노드특성
- `y=[1]` → 그래프 레벨 타겟
- `dataset.shuffle()` → 데이터셋 셔플

<br/>

다음은 


## 미니배치
## 데이터 변환

## 그래프로 학습하기
















