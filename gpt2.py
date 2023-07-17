import math
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model
import pandas as pd
import csv

df = pd.read_csv(open('data/train.tsv', errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE,encoding='windows-1252',error_bad_lines=False)

X_train = df["sentence"]
y_train = df["label"]


df = pd.read_csv(open('data/dev.tsv', errors="ignore"), delimiter="\t", quoting=csv.QUOTE_NONE,encoding='windows-1252',error_bad_lines=False)

X_test = df["sentence"]
y_test = df["label"]

MAX_LENGTH = math.ceil((X_train.apply(lambda x: len(str(x).split())).mean()))+2


PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"

# this will download and initialize the pre trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2",
    pad_token=PAD_TOKEN,
    eos_token=EOS_TOKEN,
    max_length=MAX_LENGTH,
    is_split_into_words=True)


X_train = [str(ex) + EOS_TOKEN for ex in X_train]
X_test = [str(ex) + EOS_TOKEN for ex in X_test]
X_train_ = [tokenizer(str(x), return_tensors='tf', max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True, add_special_tokens=True)['input_ids'] for x in X_train]
X_test_ = [tokenizer(str(x), return_tensors='tf', max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True, add_special_tokens=True)['input_ids'] for x in X_test]

X_train_in = tf.squeeze(tf.convert_to_tensor(X_train_), axis=1)
X_test_in = tf.squeeze(tf.convert_to_tensor(X_test_), axis=1)

X_train_mask_ = [tokenizer(str(x), return_tensors='tf', max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True, add_special_tokens=True)["attention_mask"] for x in X_train]
X_test_mask_ = [tokenizer(str(x), return_tensors='tf', max_length=MAX_LENGTH, truncation=True, pad_to_max_length=True, add_special_tokens=True)["attention_mask"] for x in X_test]

X_train_mask = tf.squeeze(tf.convert_to_tensor(X_train_mask_), axis=1)
X_test_mask = tf.squeeze(tf.convert_to_tensor(X_test_mask_), axis=1)

model = TFGPT2Model.from_pretrained("gpt2", use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
model.training = True

model.resize_token_embeddings(len(tokenizer))
for layer in model.layers:
    layer.trainable = False
    
    
input = tf.keras.layers.Input(shape=(None,), dtype='int32')
mask = tf.keras.layers.Input(shape=(None,), dtype='int32')
x = model(input, attention_mask=mask)
#x = x.last_hidden_state[:, -1]
x = tf.reduce_mean(x.last_hidden_state, axis=1)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(3, activation='softmax')(x)

clf = tf.keras.Model([input, mask], output)


base_learning_rate = 5e-4
optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
#loss=tf.keras.losses.BinaryCrossentropy()
loss=tf.keras.losses.SparseCategoricalCrossentropy()

clf.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", verbose=1, patience=3, restore_best_weights=True)
        
y_train_in = tf.constant(y_train, dtype=tf.int32)
y_test_in = tf.constant(y_test, dtype=tf.int32)

tf.config.experimental_run_functions_eagerly(True)

history = clf.fit([X_train_in, X_train_mask], y_train_in, epochs=4, batch_size=16, validation_split=0.2, callbacks=callbacks)


clf.evaluate([X_test_in, X_test_mask], y_test_in)

clf.training = False
y_pred = clf.predict([X_test_in, X_test_mask])
y_pred_out = tf.math.argmax(y_pred, axis=-1)

clf.save_weights('models/gpt2')

import seaborn as sns
from sklearn.metrics import classification_report

print(classification_report(y_test_in, y_pred_out))


with open('gptresults', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(y_pred)
