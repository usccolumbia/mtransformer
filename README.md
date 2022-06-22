# MTransformer
Materials Transformers

Ciation: Nihang Fu, Lai Wei, xxx,..., Jianjun Hu.  Materials Transformer Language Models for Generative Materials Design: a Benchmark Study. Arxiv 2022

by Machine Learning and Evolution Laboratory, University of South Carolina


### Benchmark Datasets for training inorganic materials composition transformers

ICSD-mix dataset (52317 samples)

ICSD-pure dataset (39431 samples)

Hybrid-mix dataset (418983 samples)

Hybrid-pure dataset (257138 samples)

Hybrid-strict dataset (212778 samples)

All above datasets can be downloaded from [Figshare](https://figshare.com/account/projects/142139/articles/20122796)

### Trained Materials Transformer Models

|         | ICSD-mix     | ICSD-pure | Hybrid-mix | Hybrid-pure | Hybrid-strict |
|---------|--------------|-----------|------------|-------------|---------------|
| MT-GPT     | [GPT-ICSD-mix]() |[GPT-ICSD_pure] |[GPT-Hybrid-mix] | [GPT-Hybrid-pure] | [GPT-Hybrid-strict]|
| MT-GPT2    | [GPT2-ICSD-mix]() |[GPT2-ICSD_pure] |[GPT2-Hybrid-mix] | [GPT2-Hybrid-pure] | [GPT2-Hybrid-strict]|
| MT-GPTJ    | [GPTJ-ICSD-mix]() |[GPTJ-ICSD_pure] |[GPTJ-Hybrid-mix] | [GPTJ-Hybrid-pure] | [GPTJ-Hybrid-strict]|
| MT-GPTNeo  | [GPTNeo-ICSD-mix]() |[GPTNeo-ICSD_pure] |[GPTNeo-Hybrid-mix] | [GPTNeo-Hybrid-pure] | [GPTNeo-Hybrid-strict]|
| MT-BART    | [BART-ICSD-mix]() |[BART-ICSD_pure] |[BART-Hybrid-mix] | [BART-Hybrid-pure] | [BART-Hybrid-strict]|
| MT-RoBERTa | [RoBERTa-ICSD-mix]() |[RoBERTa-ICSD_pure] |[RoBERTa-Hybrid-mix] | [RoBERTa-Hybrid-pure] | [RoBERTa-Hybrid-strict]|


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
