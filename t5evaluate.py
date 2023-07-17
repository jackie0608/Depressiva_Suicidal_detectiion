from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained('models/t5/t5', local_files_only=True).to('cuda')


import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob
from datasets import load_dataset
import datasets
from datasets import Dataset
import csv
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# Metric
from datasets import load_metric
train_df = pd.read_csv(open('data/train.tsv', errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE,encoding='windows-1252',error_bad_lines=False)
test_df = pd.read_csv(open('data/dev.tsv', errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE,encoding='windows-1252',error_bad_lines=False)

train_dataset = Dataset.from_dict(train_df)
test_dataset = Dataset.from_dict(test_df)
dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})


samples_number = len(dataset['test'])
progress_bar = tqdm(range(samples_number))
predictions_list = []
labels_list = []

for i in range(samples_number):
	torch.cuda.empty_cache()
    
	text = dataset['test']['sentence'][i]

	inputs = tokenizer.encode_plus("sst2 sentence: "+ text, padding='max_length', max_length=512, return_tensors='pt').to('cuda')
	outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
	prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

	if prediction == 'positive' or prediction == 0 or prediction == '0':
		predictions_list.append('0')
	else:
		predictions_list.append('1')

	labels_list.append(dataset['test']['label'][i])

	progress_bar.update(1)

str_labels_list = []
for i in range(len(labels_list)): str_labels_list.append(str(labels_list[i]))

from sklearn.metrics import classification_report


report = classification_report(str_labels_list, predictions_list, zero_division=0)
print(report)

with open('t5results2', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(predictions_list)