
import time

import numpy as np
import tushare as ts

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import RMSprop
from keras.utils import np_utils

from Config import SequenceLen

from StockProcess import Stock, gen_train_data

def _split(sample):
    series_sample = []
    for i in range(len(sample) - SequenceLen+1):
        window_data = simple_normalize(sample[i: i+ SequenceLen-1])
        # window_data = sample[i: i+ SequenceLen-1]
        series_sample.append(window_data)
    # print len(series_sample)
    series_sample = np.asarray(series_sample)
    return series_sample

def _one_hot(label):
    if label == 0:
        return (1,0,0)
    if label == 1:
        return (0,1,0)
    if label == 2:
        return (0,0,1)

def dataset_split(x, y, ratio = 0.3):
    x = map(_split, x)
    x = np.asarray(x)
    y = np.asarray(y)
    train = []
    [train.extend(sample) for sample in x]
    labels = []
    [labels.extend(label) for label in y]
    labels = map(lambda x:x+1, labels)
    labels = map(_one_hot, labels)
    X_train, X_test, Y_train, Y_test = train_test_split(train, labels, test_size=ratio, random_state=0)
    X_train, Y_train, X_test, Y_test = np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test)
    print 'length of samples is %s'%(len(X_train))
    print 'timesteps of samples is %s'%(len(X_train[0]))
    print 'dim of sample is %s'%(len(X_train[0][1]))
    print 'length of target is %s'%(len(Y_train))
    print hasattr(X_train[0], 'shape')
    print 'label is like %s'%(Y_train[0])
    return X_train, Y_train, X_test, Y_test

def simple_normalize(window_data):
    window_data = np.asarray(window_data)
    mean_data = np.mean(window_data, axis=0)
    normalize_data = [v-mean_data for v in window_data]
    return normalize_data

def load_data(feature, label):
    x = np.load(feature)
    y = np.load(label)
    X_train, Y_train, X_test, Y_test = dataset_split(x, y)
    return X_train, Y_train, X_test, Y_test

def build_model(layers):
    model = Sequential()

    # model.add(TimeDistributedDense(4, input_dim=layers[0], Activation='softmax'))

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        init='orthogonal',
        return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(
        output_dim=layers[2],
        init='orthogonal',
        return_sequences=False))
    model.add(Dropout(0.5))

    # model.add(LSTM(
    #     output_dim=layers[3],
    #     init='svd',
    #     return_sequences=False
    # ))
    # model.add(Dropout(0.5))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("softmax"))

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start
    return model

def test_train(model, x_train, y_train, x_test, y_test):
    print 'length of y %s'%len(y_train)
    print 'dim of y %s'%len(y_train[0])
    model.fit(x_train, y_train, batch_size=128, nb_epoch=10, verbose=1, validation_split=0.05)

    score = model.evaluate(x_test, y_test, verbose=1, show_accuracy=True)

    print score

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data('hs_300_feature_1.npy', 'hs_300_label_1.npy')
    model = build_model([14, 50, 100, 3])
    test_train(model, x_train, y_train, x_test, y_test)