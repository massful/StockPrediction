
import time

import pickle
import numpy as np
import tushare as ts

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop
from keras.utils import np_utils

from Config import SequenceLen

from StockProcess import Stock, gen_train_data

SEED = 0
np.random.seed(SEED)

class Model(object):
    def __init__(self, name):
        self.name = name

    def accuracy(self):
        ''


class Network(Model):
    def __init__(self):
        super(Network, self).__init__()

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
    std_data = np.std(window_data, axis=0)
    normalize_data = [(v-mean_data) for v in window_data]
    # normalize_data = map(lambda x:[v/std_data[i] for i,v in enumerate(x)], normalize_data)
    return normalize_data

def load_data(feature, label):
    x = np.load(feature)
    y = np.load(label)
    X_train, Y_train, X_test, Y_test = dataset_split(x, y, 0.2)
    return X_train, Y_train, X_test, Y_test

def build_model(dropout_rate=0.45, learn_rate=0.001):
    model = Sequential()

    # model.add(TimeDistributedDense(4, input_dim=layers[0], Activation='softmax'))

    '''current best'''
    model.add(LSTM(
        input_dim=14,
        output_dim=64,
        init='glorot_uniform',
        inner_init = 'orthogonal',
        forget_bias_init='one',
        return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(
        output_dim=256,
        init='glorot_uniform',
        inner_init='orthogonal',
        forget_bias_init='one',
        return_sequences=False))
    model.add(Dropout(dropout_rate))
    '''end'''

    # model.add(LSTM(
    #     output_dim=16,
    #     init='orthogonal',
    #     return_sequences=False
    # ))
    # model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=3))
    model.add(Activation("softmax"))
    optimizer = RMSprop(lr=learn_rate)

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print "Compilation Time : ", time.time() - start
    return model

def param_search(x_train, y_train, x_test, y_test):
    # epochs = [5, 10, 20]
    batch_size = [64, 128, 256]
    dropout_rate = [0.4, 0.5, 0.6]
    learn_rate = [0.001, 0.01]
    # param_dict = {'batch_size':batch_size, 'dropout_rate':dropout_rate}
    params_result = []
    for batch in batch_size:
        for dropout in dropout_rate:
            for lr in learn_rate:
                model = build_model(dropout, lr)
                model.fit(x_train, y_train, batch_size=batch, nb_epoch=10, verbose=1, validation_split=0.1)
                score = model.evaluate(x_test, y_test, verbose=1, show_accuracy=True)
                params_result.append({'batch_size':batch, 'dropout_rate':dropout, 'score':score})
    return params_result
    # grid = GridSearchCV(estimator=model, param_grid=param_dict, verbose=1, n_jobs=-1)
    # grid_result = grid.fit(x_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # for params, mean_score, scores in grid_result.grid_scores_:
    #     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

def test_train(x_train, y_train, x_test, y_test):
    print 'length of y %s'%len(y_train)
    print 'dim of y %s'%len(y_train[0])
    model = build_model(learn_rate=0.01)
    # param_result = param_search(x_train, y_train, x_test=x_test, y_test=y_test)
    # print param_result
    # pickle.dump(open('params.pkl', 'wb'), param_result)
    model.fit(x_train, y_train, batch_size=128, nb_epoch=10, verbose=1, validation_split=0.05)
    # epochs = [5, 10, 20]
    # batch_size = [32, 64, 128, 256, 512]
    # #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam']
    # learn_rate = [0.001, 0.01, 0.1]
    # #momentum = [0.0, 0.2, 0.4, 0.5]
    # init_mode = ['uniform', 'lecun_uniform', 'normal',  'glorot_normal', 'glorot_uniform', 'he_normal',
    #              'he_uniform']
    # dropout_rate = [0.2, 0.4, 0.6]
    #
    # param_dict = dict(batch_size=batch_size, nb_epoch=epochs,  learn_rate=learn_rate,  dropout_rate=dropout_rate)
    # param_search(model, param_dict, x_train, x_train)

    score = model.evaluate(x_test, y_test, verbose=1, show_accuracy=True)

    print score

    # model.save('two_layer.h5')

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data('hs_300_feature_1.npy', 'hs_300_label_1.npy')
    #model = build_model([14, 50, 100, 3])
    #model = KerasClassifier(build_fn=build_model, verbose=1, validation_split=0.1)

    test_train(x_train, y_train, x_test, y_test)