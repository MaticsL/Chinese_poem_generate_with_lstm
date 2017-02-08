from __future__ import print_function
import random
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout, Embedding
from keras.optimizers import RMSprop

from utils import init_stop_words, get_training_data, check_rythm
from constant import wu_yan_lv_shi


# init
iter_num = 200

text = get_training_data()
stop_words = init_stop_words()
print('stop_words:', stop_words)

for i in stop_words:
    text = text.replace(i, '')
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
txt_maxlen = 15000
maxlen = 5
step = 3
vec_size = 100

sentences = []
next_chars = []
for i in range(0, min(len(text), txt_maxlen) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen), dtype='float32')
y = np.zeros((len(sentences), len(chars)), dtype='bool')
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_indices[char]
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=len(chars), output_dim=vec_size,
                    input_length=maxlen))
model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.15))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.03)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=0.7):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    preds = np.exp(preds)
    preds = preds / np.sum(preds)
    p = np.argmax(preds)
    return p


# train the model, output generated text after each iteration
for iteration in range(1, iter_num):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    if iteration == 30:
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer)

    model.fit(X, y, batch_size=128, nb_epoch=1, verbose=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    for line in wu_yan_lv_shi:
        for p in line:
            x = np.zeros((1, maxlen))
            for t, char in enumerate(sentence):
                x[0, t] = char_indices[char]
            preds = model.predict(x, verbose=0)[0]
            success = False
            while not success:
                next_index = sample(preds)
                next_char = indices_char[next_index]
                success = check_rythm(p, next_char)
                preds[next_index] = 0

            generated += next_char
            sentence = (sentence + next_char)[-maxlen:]
            sys.stdout.write(next_char)
            sys.stdout.flush()
        sys.stdout.write(', ')
        sys.stdout.flush()
    print()
