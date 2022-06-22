# MTransformer
Materials Transformers

Ciation: Nihang Fu, Lai Wei, xxx,..., Jianjun Hu.  Materials Transformer Language Models for Generative Materials Design: a Benchmark Study. Arxiv 2022

by Machine Learning and Evolution Laboratory, University of South Carolina


### Benchmark Datasets for training inorganic materials composition transformers

#### ICSD-mix dataset (xxx samples)

#### ICSD-pure dataset

#### Hybrid-mix dataset

#### Hybrid-pure dataset

#### Hybrid-strict dataset

All above datasets can be downloaded from [Figshare](https://figshare.com/account/projects/142139/articles/20122796)

### Trained Materials Transformer Models

|         | ICSD-mix     | ICSD-pure | Hybrid-mix | Hybrid-pure | Hybrid-strict |
|---------|--------------|-----------|------------|-------------|---------------|
| MT-GPT     | GPT-ICSD-mix |           |            |             |               |
| MT-GPT2    |              |           |            |             |               |
| MT-GPTJ    |              |           |            |             |               |
| MT-GPTNeo  |              |           |            |             |               |
| MT-BART    |              |           |            |             |               |
| MT-RoBERTa |              |           |            |             |               |


### How to train with your own dataset

#### Installation


#### Data preparation


#### Training

Train MT-GPT materials transformers 
```
python MT-GPT-train.py --data xxxx --modelfile xxxx.pickle
```

### How to generate new materials compositions/formula using the trained models

Generate materials formulas using the trained MT-GPT 
```
python MT-GPT-generate.py --model xxxx --outputfile xxxx.csv
```
