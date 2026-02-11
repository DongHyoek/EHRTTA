# Here is for EHR TTA via using LLM with PETM 


*My plan follows the architectures described below. Let's go!*

![image](./images/blueprint.png)

# Overall Architecture
![image](./images/architecture.png)

# Embedding 순서 
1) DataEmbedding_ITS_Ind_VarPrompt로 time series 데이터 serialization하게 만들어서 임베딩
2) Tokenized Text max_length, Timeseries max_length 설정
3) 시계열 길이가 너무 적은 거 같긴 함.. 

# Aligning 과정
- (수정사항) Cross Attention에서 gate를 아예 사용하지 않고 학습해보기. 

# Impelementation orders
```
huggingface-cli login  # insert your tokens

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