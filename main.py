#%%
#Author: Alibek Zhakubayev

#Import modules
import pandas as pd
import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import TimeDistributed, LSTM, RepeatVector, Embedding, Dense
from keras.utils.vis_utils import plot_model
from unicodedata import normalize
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import csv

words = pd.read_table('glove.6B.300d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

#%%
#load Train and Test sentences
X = []
with open('sentencesTrain.txt') as file:
    lines = file.readlines()
    for line in lines:
        l = line.replace('\n', '').split(' ')
        ar = []
        for i in l:
            temp = normalize('NFD', i).encode('ascii', 'ignore')
            temp = temp.decode('UTF-8')
            temp = temp.lower().translate(str.maketrans('', '', string.punctuation))
            ar.append(temp)
        X.append(ar)
with open('sentencesTest.txt') as file:
    lines = file.readlines()
    for line in lines:
        l = line.replace('\n', '').split(' ')
        ar = []
        for i in l:
            temp = normalize('NFD', i).encode('ascii', 'ignore')
            temp = temp.decode('UTF-8')
            temp = temp.lower().translate(str.maketrans('', '', string.punctuation))
            ar.append(temp)
        X.append(ar)
#%%
#Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
mValX = 0
for i in X:
    mValX = max(mValX, len(i))
X = pad_sequences(X, maxlen=mValX, padding='post')
X = np.asarray(X)
X_test = X[702:]
X = X[:702]
#%%
#get glove representation of the words
word_embeddings = []
for i in range(np.unique(X).shape[0]):
    w = tokenizer.sequences_to_texts([[i]])[0]
    try:
        val = words.loc[w]
    except KeyError:
        val = [0] * 300
        val = np.asarray(val)
    word_embeddings.append(val)

word_embeddings = np.asarray(word_embeddings)
#%%
#Get train tokens
y = []
with open('tokensTrain.txt') as file:
    lines = file.readlines()
    for line in lines:
        temp = line.replace('\n', '').split(',')
        y.append(temp)
#%%
#Tokenizer for the target variable
tokenizer2 = Tokenizer(lower = False)
tokenizer2.fit_on_texts(y)
y = tokenizer2.texts_to_sequences(y)
mValy = 0
for i in y:
    mValy = max(mValy, len(i))
y = pad_sequences(y, maxlen=mValy, padding='post')
leny = np.unique(np.asarray(y)).shape[0]
ar = []
for i in y:
    temp = []
    for j in i:
        row = [0] * leny
        row[j] = 1
        temp.append(row)
    ar.append(temp)
y = np.asarray(ar)

#%%
#The model
model = Sequential()
model.add(Embedding(np.unique(X).shape[0], 300, weights=[word_embeddings], input_length=mValX))
model.add(LSTM(50))
model.add(RepeatVector(mValy))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(leny, activation='softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy')
#%%
#Train the model
model.fit(X, y, epochs=3000, verbose=2)
#%%
#Get predictions
pred = model.predict(X_test)
#%%
#Convert softmax output to tokens
ar = []
for i in pred:
    temp = []
    for j in i:
        temp.append(np.argmax(j))
    ar.append(temp)
ar = np.asarray(ar)
#%%
#Tokens to words
all = []
for i in ar:
    temp = i[i != 0]
    ar = []
    for k in temp:
        ar.append(tokenizer2.sequences_to_texts([[k]])[0])
    all.append(ar)
#%%
#Write the output to the file
for i in all:
    with open('tokensTest.txt', 'a') as f:
        st = ''
        for k in i:
            st = st + k + ','
        f.write(st[:-1])
        f.write('\n')