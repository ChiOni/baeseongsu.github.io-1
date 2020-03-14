---
title: 예제를 통해 알아보는 PyTorch Geometric 5 Basic Concepts
date: 2020-03-09 12:06:00
categories: [Deep Learning, Tutorial]
tags: [Deep Learning, Graph Neural Network, PyTorch]
use_math: true
seo:
  date_modified: 2020-03-11 03:25:56 +0900
---



다음 글은 [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) 라이브러리 설명서에 있는  [Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#) 를 참고하여 작성했습니다.

<br/>

<img src="https://pytorch-geometric.readthedocs.io/en/latest/_static/pyg_logo_text.svg" width="400" height="400">

<br/>

최근 Graph Neural Network에 대해 관심이 많아 공부하던 도중  <kbd>PyTorch Geometric</kbd>이라는 라이브러리를 알게 되었습니다. 실제 코드를 작성해보지 않으면, 평생 사용할 수 없을 것 같아 해당 라이브러리의 Docs를 번역하여 글을 작성했습니다. 여러 예제를 통해 **PyTorch Geometric 라이브러리에서 등장하는 5가지 기본 개념**에 대해 살펴보겠습니다.

<br/>

- TOC 
{:toc}
<br/>

---

<br/>



## **그래프의 데이터 핸들링**

**그래프**는 **노드**(node 또는 vertex)와 **그 노드를 연결하는 간선**(edge)을 하나로 모아 놓은 자료구조입니다. 그래프를 구성하는 노드와 엣지들을 모아놓은 집합을 각각 $V, E$ 라고 했을 때, 그래프는 $G=(V,E)$ 로 표현할 수 있습니다.

**PyTorch Geometric** 에서 하나의 그래프는 `torch_geometric.data.Data` 라는 클래스로 표현됩니다.  
특히, 이 클래스는 다음과 같은 default 속성이 있습니다.

- `data.x` : 노드 특징 행렬
  - [num_nodes, num_node_features]
- `data.edge_index` : 그래프의 연결성
  - [2, num_edges]
- `data.edge_attr` : 엣지 특징 행렬
  - [num_edges, num_edge_features]
- `data.y` : 학습하고 싶은 대상 (타겟)
  - 그래프 레벨 → [num_nodes, *]
  - 노드 레벨 → [1, *]
- `data.pos` : 노드 위치 행렬
  - [num_nodes, num_dimensions]

<br/>

사실 위에 있는 속성들은 필수가 아니라 **옵션**입니다. 즉, 자신이 구성하고 싶은 속성들을 다양하게 모델링할 수 있습니다. 하지만 일반적으로 그래프는 노드와 엣지로 표현하기 때문에 위의 속성들로 표현하는 것이 일반적입니다.

기존의 `torchvision`은 이미지와 타겟으로 구성된 튜플 형태로 데이터를 정의했습니다.  
그와 다르게,  `torch_geometric`은 조금 더 그래프에 직관적인 형태의 데이터 구조로 되어 있습니다.

<br/>

`Data` 클래스를 사용해 하나의 그래프 데이터를 만들어보겠습니다.

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor(([-1], [0], [1], dtype=torch.float))

data = Data(x=x, edge_index=edge_index)
>>> Data(edge_index=[2, 4], x=[3, 1])
```
- `edge_index` : (2,4) 크기의 행렬 → 4개의 엣지들
- `x` : (3,1) 크기의 행렬 → 3개의 노드와 각 노드의 특징은 단일 값

<br/>

일반적으로 엣지는 노드의 순서쌍으로 나타내는 경우가 많습니다.  
따라서 (v1, v2) 와 같은 자료형 구조가 익숙할 때가 많습니다.  
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
    <figcaption><center>위에서 정의한 data 인스턴스의 실제 그래프</center></figcaption>
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

그래프론에서 자주 사용하는 개념인 루프, 고립된 노드, 방향성 등 그래프 특징을 반영한 함수들이 있네요.  
소스코드를 뜯어보면, 각각 어떻게 동작하는지 확인할 수 있겠죠?

<br/>

<br/>

<br/>

## **공통 벤치마크 데이터셋**

**PyTorch Geometric** 은 **다양한 공통 벤치마크 데이터셋**을 포함하고 있습니다.  
해당 데이터셋의 종류는 [torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) 에서 확인할 수 있습니다.

각 데이터셋마다 그래프 데이터의 속성이 다르기 때문에 사용되는 함수가 다를 수 있습니다.  
그래프 하면 빠질 수 없는 데이터셋인, **ENZYMES** 과 **Cora** 에 대한 예시를 살펴보겠습니다.

<br/>

다음은 ENZYMES 데이터셋을 불러오는 예제입니다.

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
수많은 그래프 중 일부만 가져오고 싶을 때는 어떻게 하면 될까요?  
**인덱스 슬라이싱**을 통해 자유롭게 가져오면 됩니다.

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

- `edge_index=[2, 168]` → 총 84개의 엣지
- `x=[37, 3]` → 총 37개의 노드와 3개의 노드 특성
- `y=[1]` → 그래프 레벨 타겟
- `dataset.shuffle()` → 데이터셋 셔플

<br/>

다음은 Cora 데이터셋을 불러오는 예제입니다.  
Cora 데이터셋은 주로 (semi-supervised) graph node classification을 위한 데이터셋으로 사용됩니다.

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
>>> Cora()

len(dataset)
>>> 1

dataset.num_classes
>>> 7

dataset.num_node_features
>>> 1433
```

- `Cora()` : 데이터셋 전체가 하나의 그래프
- `num_classes` : 클래스 수 (그래프가 아니라 노드임을 알 수 있음)
- `num_node_features` : 1433개의 노드특성

<br/>

앞에서 봤던 ENNZYMES 과 다르게, Cora 데이터셋은 조금 다른 속성을 갖고 있습니다.

```python
data = dataset[0]
>>> Data(edge_index=[2, 10556], test_mask=[2708],
         train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

data.is_undirected()
>>> True

data.train_mask.sum().item()
>>> 140

data.val_mask.sum().item()
>>> 500

data.test_mask.sum().item()
>>> 1000
```

- `data = dataset[0]` : `slicing` 을 통해 그래프가 아닌 노드 하나를 가져옵니다.
- `train_mask` : 학습하기 위해 사용하는 노드들을 가리킴
- `val_mask` : 검증 시 사용하는 노드들을 가리킴
- `test_mask` : 테스트 시 사용하는 노드들을 가리킴

<br/>

<br/>

<br/>

## **미니배치**

신경망은 보통 배치단위로 학습과정이 이루어집니다. **PyTorch Geometric**은 **sparse block diagonal adjacency matrices**를 통해 미니배치 형태로 만들고, 이를 병렬화처리를 수행합니다. 이 때, 특징(feature)행렬과 타겟(target)행렬도 노드 기준 동일한 형태로 구성합니다. 그림으로 확인하면 조금 더 이해하기 쉽습니다.

<br/>

<div>
  \begin{split}\mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}\end{split}
</div>

<br/>

위의 그림과 같이 구획화된 행렬($A_i,X_i,Y_i, \ i  \in \\{1,...,n \\}$)이 하나의 배치가 되어 동작하게 됩니다.

기존 `torch`에서는 `torch.utils.data.DataLoader`를 통해 배치 단위로 데이터를 처리합니다. 
비슷하게 `torch_geometric` 에서는 `torch_geometric.data.DataLoader`를 통해 그래프 단위 데이터를 처리하게 됩니다.

<br/>

다음은 `dataloader`를 구성하는 예제입니다.


```python
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    batch
    >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    batch.num_graphs
    >>> 32
```

- `torch_geometric.data.Batch`는 `torch_geometric.data.Data`를 상속받고, `batch`라는 추가 속성을 갖습니다.
  - `batch`는 각 노드들을 각각의 배치와 매핑하는 열벡터입니다.
  - ${\begin{bmatrix} 0 & \cdots & 0 & 1 & \cdots & n - 2 & n -1 & \cdots & n - 1 \end{bmatrix}}^{\top}$처럼 표현됩니다.
  - 즉, 1082개의 노드에 대해 32개의 배치를 부여합니다.

<br/>

<br/>

<br/>

## **데이터 변환**

`torchvision`에서는, `torchvision.transforms`의 여러 함수를 사용해 (이미지) 데이터 변환을 손쉽게 할 수 있었습니다. `torch_geometric`도 마찬가지로 `torch_geometric.transforms`을 통해 **같은 패러다임을 유지**합니다. 또한 `torchvision`과 동일하게, `torch_geometric.transfomrs.Compose`을 통해 다양한 변환함수들을 손쉽게 구성할 수 있습니다. 다만, 앞에서 살펴봤던 `Data`객체(그래프 데이터)에 특화된 변환함수라고 생각하면 되겠습니다.

<br/>

다음은 ShapeNet dataset을 활용해 transforms를 적용한 예제입니다.  
먼저, ShapeNet dataset을 불러옵니다.

```python
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])

dataset[0]
>>> Data(pos=[2518, 3], y=[2518])
```

- ShapeNet은 17,000건의 3D 형태의 점 구름(point clouds) 데이터를 포함합니다. (총 16개의 카테고리로 구성)
- `pos=[2518, 3]` : 2518개의 점 데이터와 3차원임을 나타냅니다.
- 현재 `edge_index`가 없습니다. 즉, 해당 `dataset`은 연결 관계가 없습니다.

<br/>

이제 그래프 데이터를 변환해보겠습니다.

```python
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                    pre_transform=T.KNNGraph(k=6),
                    transform=T.RandomTranslate(0.01))

dataset[0]
>>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])
```

- `pre_transform = T.KNNGraph(k=6)` : KNN을 통해 데이터를 그래프 형태로 변형합니다.
  - 결괏값으로 `edge_index`가 추가된 것을 확인할 수 있습니다. (즉, 연결상태 생성)
  - `pre_transform`은 디스크 메모리에 할당하기 전에 적용합니다.
- `transform = T.RandomTranslate(0.01)` : 각 노드의 위치를 아주 조금 이동시킵니다.
  - 일종의 perturbation 작업이라고 보시면 됩니다.

<br/>

<br/>

<br/>

## **그래프로 학습하기**

앞에서 다룬 내용을 정리하면 다음과 같습니다!

- 그래프 데이터 핸들링하기
- `dataset`, `dataloader` 생성하기
- `transforms`를 사용해 데이터를 변환하기

<br/>

이제 실제 **Graph Neural Network**를 구성해보겠습니다. 

다음은 간단한 GCN layer를 구성한 뒤,  Cora 데이터셋에 적용하는 예제입니다.  
(해당 Task는 <u>Graph node classification</u>입니다.)

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
>>> Cora()
```

- Cora 데이터셋은 2708개의 "scientific publications"로 구성된 데이터입니다.
- 하나의 논문은 다른 논문들을 인용할 수 있는데, 이를 연결구조로 표현한 것이 바로 **Citation Network** 입니다.
- **Citation Network**를 하나의 그래프로 본다면, 각 **논문**은 **노드**로 볼 수 있고 **인용 관계**는 **엣지**가 됩니다.
- 또한, 논문에서 등장하는 1433개의 특정단어를 모아 하나의 단어 사전으로 만들고, 논문마다 (단어 사전에 있는) 단어들의 등장 여부를 feature vector로 만들어줌으로써 노드의 특징을 만들어줍니다.
- 해당 데이터셋에 적용할 **graph node classification**은, 임의의 논문에 대해 **논문 내 등장한 단어들과 인용 관계만을 통해 어떤 종류의 논문인지 맞히는 task**입니다.

<br/>

먼저, Graph Neural Network를 생성합니다.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```

- 2개의 `GCNConv` layer를 사용합니다. 

<br/>

학습 과정부터는 기존 `pytorch`와 상당히 유사합니다.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

- 이미 정의된 `train_mask`를 사용해 학습 데이터를 구분합니다.
- `dataloader`를 정의할 때, `train_mask`를 사용해서 구현할 수도 있습니다.

<br/>

모델을 평가합니다.

```python
model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
>>> Accuracy: 0.8150
```

- 마찬가지로, `test_mask`를 사용해 평가 데이터를 구분합니다.

<br/>

---

<br/>

**그래프**는 여러 **노드**와 **그 연결 관계**로 표현되는 추상체입니다. 사실 우리가 겪는 대부분의 상황은 그래프로 표현할 수 있을지도 모릅니다. 예를 들어, 요즘 발생하는 **코로나**(:mask:)도 그래프로 표현할 수 있습니다.  

- `node` : 사람
- `edge` : 접촉 관계

확진 여부를 `edge`의 `label`로 담아낸다면 다음과 같이 모델링할 수 있겠죠? 

- `node_features` : 사람의 특징(연령, 성별...)
- `edge_features` : 접촉 시 상황(접촉 장소의 특징, 접촉 시 주변 사람의 수...)
- `edge_label` : 확진 여부

이젠 binary node classification 문제가 되어버렸습니다. 참 쉽죠? :open_mouth: 

<br/>

GNN의 역사는 그렇게 오래되진 않았으나, 매우 많은 연구 논문이 나왔고 실제 적용한 사례들도 많습니다.  
(다 재밌는데 언제 다 읽죠...ㅠㅠㅠㅠ) 저는 딥러닝 스터디할 때, 간단한 GCN 구조를 `pytorch`로 구현한 기억이 있었는데요. 오늘 소개한 라이브러리 `torch_geometric`을 사용한다면, 확실히 좀 더 쉬울 것 같습니다 :smiley:

<br/>

제가 생각한  <kbd>PyTorch Geometric</kbd>의 **강점**은 크게 2가지 입니다.

:heavy_check_mark: `torch_geometric.nn`을 통해 다양한 레이어들을 빠르게 구성할 수 있다.

- 크게는 Convoluional Layer와 Pooling Layer로 구성
- 최신 GNN 논문들의 테크닉이 빠르게 적용됨

:heavy_check_mark: `torch_geometric.datasets`을 통해 다양한 벤치마크 데이터셋을 쉽게 이용할 수 있다.

- 각 그래프 데이터셋마다 속성과 특징이 다른 것을 잘 구현함
- 이는 `torch_geometric.data`와 연관되어 있어 그래프를 빠르게 살펴볼 수 있음

<br/>

오늘은 라이브러리의 핵심 개념에 대해서만 살펴봤습니다.  
다음에 기회가 된다면, 더 다양한 코드 예제들로 글을 한 번 작성해봐야겠네요. :sparkles:





















