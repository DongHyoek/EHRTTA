# Here is for EHR TTA via using LLM with PETM 


*My plan follows the architectures described below. Let's go!*

![image](./images/blueprint.png)

# Overall Architecture
![image](./images/architecture.png)

# Embedding 순서 
1) DataEmbedding_ITS_Ind_VarPrompt로 time series 데이터 serialization하게 만들어서 임베딩
2) Tokenized Text max_length, Timeseries max_length 설정

# Impelementation orders
```
huggingface-cli login  # insert your tokens

```