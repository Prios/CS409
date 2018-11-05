import sys, os, re, csv, codecs, numpy as np, pandas as pd
# import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from IPython.display import display

from konlpy.tag import Kkma
from konlpy.tag import Okt
from konlpy.utils import pprint

from nltk.tokenize import sent_tokenize, word_tokenize

import nltk
nltk.download('punkt')

import queue
import threading
import subprocess
import multiprocessing
from tqdm import tqdm

# Init files

train_file_name = 'train'
test_file_name = 'test'

full_dir_format = 'dataset_1/{}.txt'
processed_file_dir_format = 'dataset_1/processed_{}.txt'

train_file_dir = full_dir_format.format(train_file_name)
test_file_dir = full_dir_format.format(test_file_name)
processed_train_file_dir = processed_file_dir_format.format(train_file_name)
processed_test_file_dir = processed_file_dir_format.format(test_file_name)

# Open files

train_file = open(train_file_dir, 'r', encoding='utf8')
processed_train_file = open(processed_train_file_dir, 'r+', encoding='utf8')

test_file = open(test_file_dir, 'r', encoding='utf8')
processed_test_file = open(processed_test_file_dir, 'r+', encoding='utf8')

# Preprocess files

for line in train_file:
    processed_train_file.write(line.replace('\t', '\s', line.count('\t') - 1))

train = pd.read_csv(processed_train_file_dir, delimiter='\t', header=None, names=['comment_text', 'type'])
train = train.sample(frac=1).reset_index(drop=True)
    
for line in test_file:
    processed_test_file.write(line.replace('\t', '\s', line.count('\t') - 1))

test = pd.read_csv(processed_test_file_dir, delimiter='\t', header=None, names=['comment_text', 'type'])
test = test.sample(frac=1).reset_index(drop=True)

# concat = pd.concat([train, test], axis=0, keys=['train','test'])
# concat_test = pd.concat([train[:100], test[:50]], axis=0, keys=['train','test'])

# # Change categorical to index codes
# concat['type'] = pd.Categorical(concat['type'])
# concat_test['type'] = pd.Categorical(concat_test['type'])

# # Y is in [0, 1] where 1 is spam and 0 is normal
# y_concat = concat['type'].astype('category').cat.codes
# y_concat_test = concat_test['type'].astype('category').cat.codes

# Change categorical to index codes
# train = train[:10000]
# test = test[:5000]

train['type'] = pd.Categorical(train['type'])
test['type'] = pd.Categorical(test['type'])

# Y is in [0, 1] where 1 is spam and 0 is normal
y_t = train['type'].astype('category').cat.codes
y_te = test['type'].astype('category').cat.codes

# Get list of comments
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

import time
twitter = Okt()
def thread_pos_tagging(data_list, tag):
    q = queue.Queue()
    for i, v in enumerate(data_list):
        q.put((i, v))
        
    result = [None] * len(data_list)
    def _tagging(result):
        while True:
            i, v = q.get()
            result[i] = " ".join(["".join(w) for w, t in tag.pos(v)])
            q.task_done()

    cpus=multiprocessing.cpu_count() #detect number of cores
    print("Creating %d threads" % cpus)
    for i in range(cpus):
        t = threading.Thread(target=_tagging, args=(result,))
        t.daemon = True
        t.start()

    q.join()
    return result

start = time.time()
print ("> POS Tagging: Train")
# doc0_t = thread_pos_tagging(list_sentences_train, twitter)
doc0_t = []
for s in tqdm(list_sentences_train):
    doc0_t.append(" ".join(["".join(w) for w, t in twitter.pos(s)]))
# doc0_t = [" ".join(["".join(w) for w, t in twitter.pos(s)]) for s in list_sentences_train]
print('> Execution Time: %.2f' % (time.time() - start))
print ("> POS Tagging: Test")
start = time.time()
# doc0_te = thread_pos_tagging(list_sentences_test, twitter)
doc0_te = []
for s in tqdm(list_sentences_test):
    doc0_te.append(" ".join(["".join(w) for w, t in twitter.pos(s)]))
# doc0_te = [" ".join(["".join(w) for w, t in twitter.pos(s)]) for s in list_sentences_test]
print('> Execution Time: %.2f' % (time.time() - start))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(doc0_t)
list_tokenized_train = tokenizer.texts_to_sequences(doc0_t)
list_tokenized_test = tokenizer.texts_to_sequences(doc0_te)

print(len(list_tokenized_train))
print(len(list_tokenized_test))

maxlen = max([len(x) - 1 for x in list_tokenized_train])
vocab_size = len(tokenizer.word_index) + 1
print ("> Maxlen: %d" % maxlen)

# Pad sequences

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

# Input layer

inp = Input(shape=(maxlen, ))

# Embed layer

embed_size = 200
x = Embedding(vocab_size, embed_size)(inp)

# LSTM layer

x = LSTM(200, return_sequences=False, name='lstm_layer')(x)

# Dropout layer (1/2)

x = Dropout(0.1)(x)

# Dense Layer

x = Dense(50, activation="relu")(x)

# Droupout layer (2/2)

x = Dropout(0.1)(x)

# Final Dense layer

x = Dense(1, activation="sigmoid")(x)

# Optimizer

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

print(model.summary())

# Train model

batch_size = 100
epochs = 5
model.fit(X_t, y_t, batch_size=batch_size, epochs=epochs, validation_split=0.1)

scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# test_text = "못 보신 분들 \"마이티빙\"에서 무료로 보세요. 가입필요 없음."

# tokenized = tokenizer.texts_to_sequences([test_text])

# test_sequence = pad_sequences(tokenized, maxlen=maxlen)

# model.predict(test_sequence)