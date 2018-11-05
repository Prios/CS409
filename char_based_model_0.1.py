import sys, os, re, csv, codecs, numpy as np, pandas as pd
# import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Convolution1D, MaxPooling1D, concatenate
from keras.models import Model, Sequential
from keras.optimizers import SGD
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

# units
units = 100

# Convolution parameters
filter_length = 3
nb_filter = 150
pool_length = 2
cnn_activation = 'relu'
border_mode = 'same'

# RNN parameters
output_size = 50
rnn_activation = 'tanh'
recurrent_activation = 'hard_sigmoid'

# Compile parameters
loss = 'binary_crossentropy'
optimizer = 'rmsprop'

# Training parameters
batch_size = 50
nb_epoch = 3
validation_split = 0.25
shuffle = True

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
train = train[:50000]
test = test[:5000]

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
remove_indices = []
for i, l in enumerate(tokenizer.texts_to_sequences(doc0_t)):
    if len(l) > 200:
        remove_indices.append(i)
list_tokenized_train = [l for i, l in enumerate(tokenizer.texts_to_sequences(doc0_t)) if i not in remove_indices]

remove_indices = []
for i, l in enumerate(tokenizer.texts_to_sequences(doc0_te)):
    if len(l) > 200:
        remove_indices.append(i)
list_tokenized_test = [l for i, l in enumerate(tokenizer.texts_to_sequences(doc0_te)) if i not in remove_indices]

# list_tokenized_train = [l for l in tokenizer.texts_to_sequences(doc0_t) if len(l) < 200]
# list_tokenized_test = [l for l in tokenizer.texts_to_sequences(doc0_te) if len(l) < 200]

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

embed_size = 300
embedding_layer = Embedding(vocab_size, embed_size, input_length=maxlen)

#########################################
# Simple RNN
model = Sequential()
model.add(embedding_layer)
model.add(SimpleRNN(output_dim=output_size, activation=rnn_activation))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
print("> Simple RNN")
print(model.summary())
model.fit(X_t, y_t, batch_size=batch_size, epochs=nb_epoch,validation_split=validation_split,shuffle=shuffle)
scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#########################################
# GRU

model = Sequential()
model.add(embedding_layer)
model.add(GRU(output_dim=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('> GRU')
print(model.summary())
model.fit(X_t, y_t, batch_size=batch_size, epochs=nb_epoch,validation_split=validation_split,shuffle=shuffle)
scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#################################################
# Bidirectional LSTM

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(units=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation)))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('> Bidirectional LSTM')
print(model.summary())
model.fit(X_t, y_t, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)
scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
##################################################
# LSTM

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.25))
model.add(LSTM(units=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('> LSTM')
print(model.summary())
model.fit(X_t, y_t, batch_size=batch_size, epochs=nb_epoch,validation_split=validation_split,shuffle=shuffle)
scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

##########################################################
# CNN + LSTM

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode=border_mode,
                        activation=cnn_activation,
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(units=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('> CNN + LSTM')
print(model.summary())
model.fit(X_t, y_t, batch_size=batch_size, epochs=nb_epoch,validation_split=validation_split,shuffle=shuffle)
scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

###########################################################
# CNN
# Based on "Convolutional Neural Networks for Sentence Classification" by Yoon Kim http://arxiv.org/pdf/1408.5882v2.pdf
# https://github.com/keon/keras-text-classification/blob/master/train.py

filter_sizes = (3,4,5)
num_filters = 100
graph_in = Input(shape=(maxlen, embed_size))
convs = []
for fsz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)

if len(filter_sizes) > 1:
    out = concatenate(convs)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out)
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.25, input_shape=(maxlen, embed_size)))
model.add(graph)
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
opt = SGD(lr=0.01, momentum=0.80, decay=1e-6, nesterov=True)
model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
model.fit(X_t, y_t, batch_size=batch_size, epochs=nb_epoch,validation_split=validation_split,shuffle=shuffle)
scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
print(model.summary())
model.fit(X_t, y_t, batch_size=batch_size, epochs=nb_epoch,validation_split=validation_split,shuffle=shuffle)
scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# def model(x):
#     # LSTM layer
#     x = LSTM(200, return_sequences=False, name='lstm_layer')(x)
#     # Dropout layer (1/2)
#     x = Dropout(0.5)(x)
#     # Dense Layer
#     x = Dense(50, activation="relu")(x)
#     # Droupout layer (2/2)
#     x = Dropout(0.5)(x)
#     # Final Dense layer
#     x = Dense(1, activation="sigmoid")(x)
#     return x
# # Optimizer
# model = Model(inputs=inp, outputs=x)
# model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
# print(model.summary())

# # Train model

# batch_size = 100
# epochs = 5
# model.fit(X_t, y_t, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# scores = model.evaluate(X_te, y_te, verbose=1)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# test_text = "못 보신 분들 \"마이티빙\"에서 무료로 보세요. 가입필요 없음."

# tokenized = tokenizer.texts_to_sequences([test_text])

# test_sequence = pad_sequences(tokenized, maxlen=maxlen)

# model.predict(test_sequence)