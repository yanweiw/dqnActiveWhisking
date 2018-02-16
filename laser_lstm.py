#!/usr/bin/env python

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np

batch_size = 32

print('Loading data...')
x_train = np.vstack((np.load('data/tri_data_1000.npy'), np.load('data/hex_data_1000.npy')))
y_train = np.vstack((np.zeros((1000, 1)), np.ones((1000, 1))))
x_test = np.vstack((np.load('data/tri_data_200.npy'), np.load('data/hex_data_200.npy')))
y_test = np.vstack((np.zeros((200, 1)), np.ones((200, 1))))
print(len(x_train), 'train_sequences; x_train shape:', x_train.shape)
print(len(x_test), 'test_sequences; x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(LSTM(100, input_shape=(10, 19)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print('fitting...')
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

print('predicting...')
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
