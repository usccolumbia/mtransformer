# MTransformer
Materials Transformers

Ciation: Nihang Fu, Lai Wei, Yuqi Song, Qinyang Li, Rui Xin, Sadman Sadeed Omee, Rongzhi Dong, Edirisuriya M. Dilanga Siriwardane, Jianjun Hu.  Materials Transformer Language Models for Generative Materials Design: a Benchmark Study. Arxiv 2022

by < a href="http://mleg.cse.sc.edu">Machine Learning and Evolution Laboratory</a>, University of South Carolina


### Benchmark Datasets for training inorganic materials composition transformers

ICSD-mix dataset (52317 samples)

ICSD-pure dataset (39431 samples)

Hybrid-mix dataset (418983 samples)

Hybrid-pure dataset (257138 samples)

Hybrid-strict dataset (212778 samples)

All above datasets can be downloaded from [Figshare](https://figshare.com/articles/dataset/MT_dataset/20122796)

### Trained Materials Transformer Models

|         | ICSD-mix     | ICSD-pure | Hybrid-mix | Hybrid-pure | Hybrid-strict |
|---------|--------------|-----------|------------|-------------|---------------|
| MT-GPT     | [GPT-Im](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998787) |[GPT-Ip](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998787) |[GPT-Hm](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998787) | [GPT-Hp](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998787) | [GPT-Hs](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998787)|
| MT-GPT2    | [GPT2-Im](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998790) |[GPT2-Ip](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998790) |[GPT2-Hm](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998790) | [GPT2-Hp](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998790) | [GPT2-Hs](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998790)|
| MT-GPTJ    | [GPTJ-Im](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998793) |[GPTJ-Ip](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998793) |[GPTJ-Hm](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998793) | [GPTJ-Hp](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998793) | [GPTJ-Hs](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998793)|
| MT-GPTNeo  | [GPTNeo-Im](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998796) |[GPTNeo-Ip](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998796) |[GPTNeo-Hm](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998796) | [GPTNeo-Hp](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998796) | [GPTNeo-Hs](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998796)|
| MT-BART    | [BART-Im](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998784) |[BART-Ip](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998784) |[BART-Hm](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998784) | [BART-Hp](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998784) | [BART-Hs](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998784)|
| MT-RoBERTa | [RoBERTa-Im](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998799) |[RoBERTa-Ip](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998799)|[RoBERTa-Hm](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998799) | [RoBERTa-Hp](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998799) | [RoBERTa-Hs](https://figshare.com/articles/online_resource/MT_models/20123483?file=35998799)|


### How to train with your own dataset

#### Installation
1. Create your own conda or other enviroment.
2. install basic packages
```
pip install -r requirements.txt
```
3. Install `pytorch` from [pytorch web](https://pytorch.org/get-started/previous-versions/) given your python & cuda version
#### Data preparation
Download datasets from the above link, then unzip it under `MT_dataset` folder.
After the above, the directory should be:
```
MTransformer
   ├── MT_dataset
       ├── hy_mix
           ├── test.txt
           ├── train.txt
           ├── valid.txt
       ├── hy_pure
       ├── hy_strict
       ├── icsd_mix
       ├── icsd_pure
       ├── mp
   ├── MT_models
       ├── MT_Bart
           ├── hy_mix
               ├── config.json
               ├── pytorch_model.bin
               ├── training_args.bin
           ├── hy_pure
           ├── hy_strict
           ├── icsd_mix
           ├── icsd_pure
       ├── MT_GPT
       ├── MT_GPT2
       ├── MT_GPTJ
       ├── MT_GPTNeo
       ├── MT_RoBERTa
       ├── tokenizer
           ├── vocab.txt       
   ├── generateFormula_random.py
   ├── multi_generateFormula_random.py
   ├── README.md
   └── requirements.txt
```
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
