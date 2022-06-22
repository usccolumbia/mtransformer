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
| MT-GPT     | [GPT-Im](https://figshare.com/account/projects/142139/articles/20123483?file=35994485) |[GPT-Ip](https://figshare.com/account/projects/142139/articles/20123483?file=35994485) |[GPT-Hm](https://figshare.com/account/projects/142139/articles/20123483?file=35994485) | [GPT-Hp](https://figshare.com/account/projects/142139/articles/20123483?file=35994485) | [GPT-Hs](https://figshare.com/account/projects/142139/articles/20123483?file=35994485)|
| MT-GPT2    | [GPT2-Im](https://figshare.com/account/projects/142139/articles/20123483?file=35994491) |[GPT2-Ip](https://figshare.com/account/projects/142139/articles/20123483?file=35994491) |[GPT2-Hm](https://figshare.com/account/projects/142139/articles/20123483?file=35994491) | [GPT2-Hp](https://figshare.com/account/projects/142139/articles/20123483?file=35994491) | [GPT2-Hs](https://figshare.com/account/projects/142139/articles/20123483?file=35994491)|
| MT-GPTJ    | [GPTJ-Im](https://figshare.com/account/projects/142139/articles/20123483?file=35994497) |[GPTJ-Ip](https://figshare.com/account/projects/142139/articles/20123483?file=35994497) |[GPTJ-Hm](https://figshare.com/account/projects/142139/articles/20123483?file=35994497) | [GPTJ-Hp](https://figshare.com/account/projects/142139/articles/20123483?file=35994497) | [GPTJ-Hs](https://figshare.com/account/projects/142139/articles/20123483?file=35994497)|
| MT-GPTNeo  | [GPTNeo-Im](https://figshare.com/account/projects/142139/articles/20123483?file=35994503) |[GPTNeo-Ip](https://figshare.com/account/projects/142139/articles/20123483?file=35994503) |[GPTNeo-Hm](https://figshare.com/account/projects/142139/articles/20123483?file=35994503) | [GPTNeo-Hp](https://figshare.com/account/projects/142139/articles/20123483?file=35994503) | [GPTNeo-Hs](https://figshare.com/account/projects/142139/articles/20123483?file=35994503)|
| MT-BART    | [BART-Im](https://figshare.com/account/projects/142139/articles/20123483?file=35994482)) |[BART-Ip](https://figshare.com/account/projects/142139/articles/20123483?file=35994482) |[BART-Hm](https://figshare.com/account/projects/142139/articles/20123483?file=35994482) | [BART-Hp](https://figshare.com/account/projects/142139/articles/20123483?file=35994482) | [BART-Hs](https://figshare.com/account/projects/142139/articles/20123483?file=35994482)|
| MT-RoBERTa | [RoBERTa-Im](https://figshare.com/account/projects/142139/articles/20123483?file=35994506) |[RoBERTa-Ip](https://figshare.com/account/projects/142139/articles/20123483?file=35994506)|[RoBERTa-Hm](https://figshare.com/account/projects/142139/articles/20123483?file=35994506) | [RoBERTa-Hp](https://figshare.com/account/projects/142139/articles/20123483?file=35994506) | [RoBERTa-Hs](https://figshare.com/account/projects/142139/articles/20123483?file=35994506)|


### How to train with your own dataset

#### Installation
```
pip install -r requirements.txt
```

#### Data preparation
Download datasets from the above link, then unzip it under `MT_dataset` folder.

#### Training
An example is to train a MT-GPT model on the Hybrid-mix dataset. 
```
python ./MT_model/MT_GPT/train_GPT.py  --tokenizer ./MT_model/tokenizer/   --train_data  ./MT_Dataset/hy_mix/train.txt  --valid_data ./MT_Dataset/hy_mix/valid.txt  --output_dir ./output
```
The training for other models is similar to MT-GPT.

### How to generate new materials compositions/formula using the trained models
Download models from the above link or use your own trianed models, then put them into correspoding folders.

Generate materials formulas using the trained MT-GPT 
```
python generateFormula_random.py  --tokenizer ./MT_model/tokenizer  --model_name OpenAIGPTLMHeadModel  --model_path ./MT_model/MT_GPT2/hy_mix
```

We also provide multi-thread generation. The default number of threads is 10, and you can change it using arg `n_thread`.
```
python multi_generateFormula_random.py  --tokenizer ./tokenizer  --model_name GPT2LMHeadModel  --model_path ./MT_GPT2/hy_mix  --n_thread 5
```
