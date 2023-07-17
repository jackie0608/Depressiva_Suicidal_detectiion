### Create virtual environment with Python 3.9.10 binary, activate it
```
python3.9 -m venv env
source env/bin/activate
```
### 3.8也可以

### Install necessary packages
```
pip install -r requirements.txt
```
### Use Sentiment Analysis with my model

### Train your own model and use Sentiment Analysis with it

### --num_eps number of epoches default = 2

### --lr learning rate default = 2e-5

### --batch_size default = 32

### train file 需要路径及名称 /data/train.tsv

### test file 需要路径及名称 /data/dev.tsv

## Train (i.e.fine-tune) BERT/Albert/XLNet

###                                   使用的预训练模型       保存模型的文件夹名
```
python train.py --model_name_or_path bert-base-uncased --output_dir bert --num_eps 2
```
*bert-base-uncased, albert-base-v2, distilbert-base-uncased, and other similar models are supported.*

### Evaluate

###                                      保存的模型路径
```
python evaluate.py --model_name_or_path ./models/bert
```

### Analyze your inputs
```
python analyze.py --model_name_or_path ./models/bert
```

## Train (i.e.fine-tune) gpt2/t5
```
python gpt2.py
python t5.py
```
### t5 参数可在 line 132 training_arg 中修改
### t5 需求 transformers == 4.28.1

### gpt2 参数可在 line 25,46,68,83 分别修改
