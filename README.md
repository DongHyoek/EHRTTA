# Here is for EHR TTA via using LLM with PETM 


*My plan follows the architectures described below. Let's go!*

![image](./images/blueprint.png)

# Overall Architecture
![image](./images/architecture.png)

# Embedding 순서 
1) `DataEmbedding_ITS`로 time series 데이터 serialization하게 만들어서 임베딩 이때, 시계열 데이터(vital sign)들은 pooling 하지 않고 활용

2) 텍스트 데이터는 각 Demogrpahic, Lab, Output 변수 별로 만들고, Demographic은 그냥 값만 나머지는 10개의 필드 
(변수명 및 정상 범주, 첫번째 관측 시간, 마지막 관측 시간, ... )에 대해서 생성
`그러므로 4(demogrpahics) + 41 * 10 = 414개의 text들이 생성`

3) 이후 `TextEncoder`로 각 text마다 학습가능한 `[CLS]` 토큰을 앞에다가 붙여서 각 text의 내용을 요약하는 attention 학습, 이때 FFN과 Residual 구조는 사용 X. FFN이 들어가버리면 비선형 과정이 추가되어 벡터의 형태가 달라지는 위험. 

4) 시계열 데이터 : (B, V(변수 개수 = 7), ts_seq, d_model) / 텍스트 데이터 : (B, 414, d_model) / 시계열 마스크 (B, V, ts_seq)


# Aligning 과정
- (수정사항) Cross Attention에서 gate를 아예 사용하지 않고 학습해보기.
- 마찬가지로 FFN과 Residual 구조는 삭제. 마찬가지로 FFN이 들어가버리면 비선형 과정이 추가되어 LLM이 학습되었던 input vector들과 다른 이슈가 발생할 수 있기 때문.

# Modeling 과정
- Align된 vector를 토대로 DoRA가 추가된 LLM model 이후 나온 마지막 hidden stae의 마지막 토큰의 벡터를 최종 벡터로 간주.

# Impelementation orders
```
huggingface-cli login  # insert your tokens
sh scripts/train.sh    # refer to `build_args` function in main.py file.

```

# Codes Information & Organization
```
EHRTTA
├─data
│  ├─eicu
│  ├─hirid
│  └─miiv
├─images
├─models
│  ├─align.py
│  ├─embed.py
│  ├─llm.py 
│  └─tta.py
├─preprocess
│  └─ehr
├─scripts
├─utils
│  ├─dataloader.py
│  ├─inference.py
│  ├─norm.py 
│  └─trainer.py

```