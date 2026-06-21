# Agent Environments - SearchQA

## Setup

```sh
conda env create -f environment.yml
conda activate agentenv-searchqa
pip install -e .
bash ./setup.sh
```

## Launch

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 SEARCHQA_FAISS_GPU=true searchqa --host 0.0.0.0 --port 36001

CUDA_VISIBLE_DEVICES=4,5,6,7 SEARCHQA_FAISS_GPU=true searchqa --host 0.0.0.0 --port 36002
```

## Environment variables

`SEARCHQA_FAISS_GPU`: Force enable RAG server on GPUs

> Other variables please refer to `env_warpper.py` line 50-68

## Item ID

| Item ID         | Description             | Split |
| --------------- | ----------------------- | ----- |
| 0 ~ 3609        | nq Dataset              | Test  |
| 3610 ~ 14922    | triviaqa Dataset        | Test  |
| 14923 ~ 29189   | popqa Dataset           | Test  |
| 29190 ~ 36594   | hotpotqa Dataset        | Test  |
| 36595 ~ 49170   | 2wikimultihopqa Dataset | Test  |
| 49171 ~ 51587   | musique Dataset         | Test  |
| 51588 ~ 51712   | bamboogle Dataset       | Test  |
| 51713 ~ 130880  | nq Dataset              | Train |
| 130881 ~ 221328 | hotpotqa Dataset        | Train |


/mnt/data/kw/anaconda3/envs/agentenv-searchqa/lib/python3.10/site-packages/agentenv_searchqa 修改代码记得在这里放