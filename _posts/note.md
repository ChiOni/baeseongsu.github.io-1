

Conditional language mdoel

encoder-deocder architure



CNN + RNN

- encoder : CSM(Convoluitional Sentence Model)
- decoder : RNN
  - Unconditioanl RNN 과 다르게, source sentence를 bias와 함께 넣어줌
  - 

Seq-to-seq

- cell state, hidden state가 decoder의 initial state, hidden state로 들어감
- 아키텍쳐 구조를 거꾸로 하면 성능이 더 좋았다
- Beam-search 기법



위의 모델에 문제 Encoder에서 꼭 하나의 벡터로 줘야 하나? (gradient가 너무 먼길을 가야함)

LSTM의 long-term 본다해도 gradient vanishing problem 해결?

source sentence를 matrix로 표현한다면 해결할 수 있지 않을까?



그 중에 가장 좋은 성능을 내는 Bi-directional RNN

- 양방향으로 encoder로 matrix를 생성



문제? matrix를 어떻게 deocder부분에 넣어야하는가?

- attention으로





Attention is all you need

transformer is the first sequence-to-sequence relying entirely on self-attention





Seq-seq model with attention

c_i => alpha attention

decoder i 에 대해 encoder j들이 각각 어떤 영향을 가지는가?





The transformer

- RNN/CNN 사용하지 않고 self-attention만 사용
- 압도적으로 성능이 좋음
- CPU에서 최대로 병렬화 가능

- eoncoder - connection - decoder

여러 encoder를 쌓음 -> decoder를 encoder와 동일한 개수만큼 쌓음

- encoder 2개 레이어

  - SA

    - 왜 좋은가?

    - refines the representation by matching a single sequence aginst itself

    - procedure

      - 1. calculate query key value vector for each word

           With traininable projection matrices

        2. Query 와 key의 dot product (self-attention score)

           query i key j => a_ij = > i번째 단어가 j번째 단어에 두어야 하는 중요도

        3. calculate a weighted sum of the value vector

    - self attention with multi-heads (본 논문에서는 8개의 head)

    - 가로로 concat을 함 => W0로 곱해서 최종 z값이 나옴

    - positional encoding ; word의 위치정보를 넣어줌

      - 각각의 위치에 대해서 다른 벡터가 더해짐
      - 위치상의 distance가 positional encoding으로 들어갈 수 있다.
      - 학습 parameter로 해봤는데 nonlearnable이 더 좋았다.

    - residual connection이 들어감

  - FF

    - 동일한 네트워크임

  GPU의 극대화를 노림

  RNN과 다르게 time depedency가 없음

- decoder 3개 레이어

  - SA
    - mask가 들어감
    - 아직 decoding 되지 않은 정보들은 보지 않겠다.
    - 이전 정보를 보지않겠다. (time)
  - EDA
  - FF









---





graph classification task

- label missing 으로 했을 때,
- 





graph classification (semi-supervised learning)

baseline