import sys, os, re, csv, codecs, numpy as np, pandas as pd
# import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from konlpy.tag import Kkma
from konlpy.utils import pprint

train_file_name = 'train'
test_file_name = 'test'

full_dir_format = 'dataset_1/{}.txt'
processed_file_dir_format = 'dataset_1/processed_{}.txt'

train_file_dir = full_dir_format.format(train_file_name)
test_file_dir = full_dir_format.format(test_file_name)
processed_train_file_dir = processed_file_dir_format.format(train_file_name)
processed_test_file_dir = processed_file_dir_format.format(test_file_name)

train_file = open(train_file_dir, 'r+', encoding='utf8')
processed_train_file = open(processed_train_file_dir, 'r+', encoding='utf8')

test_file = open(test_file_dir, 'r+', encoding='utf8')
processed_test_file = open(processed_test_file_dir, 'r+', encoding='utf8')


for line in train_file:
    processed_train_file.write(line.replace('\t', '\s', line.count('\t') - 1))

train = pd.read_csv(processed_train_file_dir, delimiter='\t', header=None, names=['comment_text', 'type'])
train = train.sample(frac=1).reset_index(drop=True)

for line in test_file:
    processed_test_file.write(line.replace('\t', '\s', line.count('\t') - 1))

test = pd.read_csv(processed_test_file_dir, delimiter='\t', header=None, names=['comment_text', 'type'])
test = test.sample(frac=1).reset_index(drop=True)

train['type'] = pd.Categorical(train['type'])
test['type'] = pd.Categorical(test['type'])

# Y is in [0, 1] where 1 is spam and 0 is normal
y_t = train['type'].astype('category').cat.codes
y_te = test['type'].astype('category').cat.codes

# Get list of comments
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features, char_level=True)

tokenizer.fit_on_texts(list(list_sentences_train) + )
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 100
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen, ))

embed_size = 128
x = Embedding(max_features, embed_size)(inp)

x = LSTM(60, return_sequences=True, name='lstm_layer')(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_t, y_t, batch_size=batch_size, epochs=epochs, validation_split=0.1)

scores = model.evaluate(X_te, y_te, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))