#!/usr/bin/env python

# Felix Yanwei Wang @ Northwestern University MSR, Feb 2018

# from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
# from keras.layers import TimeDistributed
from keras import backend as K
from keras.models import Model
import numpy as np

print('Loading data...')
x_train = np.vstack((np.load('data/tri_data_1000.npy'), np.load('data/hex_data_1000.npy')))
y_train = np.vstack((np.zeros((1000, 1)), np.ones((1000, 1))))
x_test = np.vstack((np.load('data/tri_data_200.npy'), np.load('data/hex_data_200.npy')))
y_test = np.vstack((np.zeros((200, 1)), np.ones((200, 1))))
print(len(x_train), 'train_sequences; x_train shape:', x_train.shape)
print(len(x_test), 'test_sequences; x_test shape:', x_test.shape)

print('Building model...')
model = Sequential()
model.add(LSTM(100, input_shape=(None, 19), name='lstm'))
# model.add(Dense(64, activation='relu', name='state'))
model.add(Dense(1, activation='sigmoid', name='output'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print('Fitting...')
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15, batch_size=64)

print('Building another stateful model of batch_size = 1...')
model2 = Sequential()
model2.add(LSTM(100, batch_input_shape=(1, 1, 19), name='lstm', stateful=True))
model2.add(Dense(1, activation='sigmoid', name='guess'))
print('Copying weights...')
model2.set_weights(model.get_weights())
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# print('Validating for TimeDistributed version...')
# for i in range(1, 11):
#     x_t = x_test[[0], 0:i, :]
#     # model.reset_states()
#     # scores = model.evaluate(x_t, y_test, verbose=0)
#     print("Accuracy: %.2f%%" % (scores[1]*100))

print('Evaluating...')
model2.reset_states()
output_states = []
for i in range(0, 50):
    print(model2.evaluate(x_test[302:303, [i], :], np.ones((1,1)), batch_size=1))
    output_states.append(K.get_value(model2.layers[0].states[1])) # layer 0 is the lstm, states[0] is memory, states[1] is output
    # can also use predict
    # print(model2.predict(x_test[0:1, [i], :]))

# model3=Model(inputs=model2.input, outupts=model2.get_layer('lstm').output) # This is problematic because model2 and model3 will share the same stateful layer



print('Saving the model...')
model2.reset_states()
model2.save('models/lstm_tri_hex.h5')
