import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob
from datasets import load_dataset
import datasets
from datasets import Dataset
import csv

train_df = pd.read_csv(open('data/train.tsv', errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE,encoding='windows-1252',error_bad_lines=False)
test_df = pd.read_csv(open('data/dev.tsv', errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE,encoding='windows-1252',error_bad_lines=False)

train_dataset = Dataset.from_dict(train_df)
test_dataset = Dataset.from_dict(test_df)
dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset})

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_id="t5-base"

# Load tokenizer of FLAN-t5-base
tokenizer = T5Tokenizer.from_pretrained(model_id)

import random

dataset['train'] = dataset['train'].shuffle(seed=42)

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
dataset.clear()
train_df['label'] = train_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)
dataset['train'] = Dataset.from_pandas(train_df)
dataset['test'] = Dataset.from_pandas(test_df)



from datasets import concatenate_datasets

# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["sentence"], truncation=True), batched=True, remove_columns=['sentence', 'label'])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["label"], truncation=True), batched=True, remove_columns=['sentence', 'label'])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

def preprocess_function(sample, padding="max_length"):
    # add prefix to the input for t5
    inputs = [item for item in sample["sentence"]]
    inputs = ["sst2 sentence: "+ sub for sub in inputs]
    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["label"], max_length=max_target_length, padding=padding, truncation=True)
    
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['sentence', 'label'])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

from transformers import AutoModelForSeq2SeqLM

# huggingface hub model id
model_id="t5-base"

# load model from the hub
model = T5ForConditionalGeneration.from_pretrained(model_id)
#model = T5ForConditionalGeneration.from_pretrained('models/t5/checkpoint-306', local_files_only=True)

import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# Metric
from datasets import load_metric
metric = load_metric('f1')

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels
    
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    for i in range(len(decoded_preds)):
        if decoded_preds[i] == 'positive':
            decoded_preds[i] = 0
        else:
            decoded_preds[i] = 1
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result
    

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir="models/t5",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-4,

    num_train_epochs=2,
    # logging & evaluation strategies
    logging_dir=f"models/t5/logs",
    logging_strategy="epoch", 
    # logging_steps=1000,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    #report_to="tensorboard",
    #push_to_hub=True,
    #hub_strategy="every_save",
    #hub_model_id=repository_id,
    #hub_token=HfFolder.get_token(),
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)


# Start training 
trainer.train()

trainer.evaluate()# Save our tokenizer and create model card

tokenizer.save_pretrained(save_directory="models/t5")

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

samples_number = len(dataset['test'])
progress_bar = tqdm(range(samples_number))
predictions_list = []
labels_list = []
for i in range(samples_number):
	torch.cuda.empty_cache()
    
	text = dataset['test']['sentence'][i]
	inputs = tokenizer.encode_plus("sst2 sentence: "+ text, padding='max_length', max_length=512, return_tensors='pt').to('cuda')
	#decoder_input_ids = torch.tensor([tokenizer.pad_token_id]).unsqueeze(0) 
	#logits = model(**inputs, decoder_input_ids=decoder_input_ids)[0]
	#tokens = torch.argmax(logits,dim=2)
	outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
	prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
	#prediction = tokenizer.batch_decode(tokens)
	#logits = logits.squeeze(1)
	#selected_logits = logits[:, [1465, 2841]] 
	#probs = F.softmax(selected_logits, dim=1)
	#predictions_list.append(prediction)


    # no idea why it sometimes give string, sometimes give number
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

with open('t5results', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(predictions_list)
