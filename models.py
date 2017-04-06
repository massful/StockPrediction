
import time

import pickle
import numpy as np
import tushare as ts

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score, accuracy_score, recall_score

from sklearn.externals import joblib

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, merge

from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.convolutional import Convolution1D,MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l1, l2, activity_l2
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils

from Config import SequenceLen

from StockProcess import Stock, gen_train_data

SEED = 0
np.random.seed(SEED)

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

def _fft(sample):
    return np.fft.fft(sample,n=64, norm='ortho')

def reshape_fft(x):
    reshape_samples = []
    for sample in x:
        sample = np.array(sample).transpose()
        tmp = map(_fft, sample)
        tmp_reshape = []
        [tmp_reshape.extend(row.real) for row in tmp]
        [tmp_reshape.extend(row.imag) for row in tmp]
        reshape_samples.append(tmp_reshape)
    return reshape_samples

def normalize_2d(x_train, x_test):
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    return x_train, x_test

def _metrics(y, pred):
    accuracy = accuracy_score(y, pred)
    recall = recall_score(y, pred, average = None)
    f1score = f1_score(y, pred, average = None)
    return 'accuracy : %s\nrecall : %s\nf1score : %s\n'%(accuracy, recall, f1score)

def reshape_time(x):
    reshape_samples = []
    for sample in x:
        tmp_reshape = []
        [tmp_reshape.extend(row) for row in sample]
        reshape_samples.append(tmp_reshape)
    return reshape_samples

def fft_svm(x, y, x_test, y_test, freq):
    train = reshape_fft(x)
    train, label = np.asarray(train), np.asarray(y)
    x_test = reshape_fft(x_test)
    x_test, y_test = np.asarray(x_test), np.asarray(y_test)
    train, x_test = normalize_2d(train, x_test)

    clf1 = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=500, n_jobs=-1, verbose=10)
    #clf2 = SVC(verbose=1)
    clf1.fit(train, label)
    #clf2.fit(train, label)
    #joblib.dump(clf1, 'softmax_'+freq+'.m')
    #joblib.dump(clf2, 'svm_'+freq+'.m')
    #print 'softmax result: %s'%clf1.score(x_test, y_test)
    pred = clf1.predict(x_test)
    print 'metrics : %s'%(_metrics(y_test, pred))
    #print 'svm result: %s'%clf2.score(x_test, y_test)

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

    # model.add(Convolution1D(input_dim=14, nb_filter=32, filter_length=3, activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(Convolution1D(nb_filter=64, filter_length=3, activation='relu'))
    # model.add(MaxPooling1D())

    '''current best'''
    model.add(LSTM(
        # input_dim=14,
        output_dim=128,
        init='glorot_uniform',
        inner_init = 'orthogonal',
        forget_bias_init='one',
        W_regularizer=0.0001,
        return_sequences=True))
    model.add(Dropout(0.5))

    # model.add(PReLU())

    model.add(LSTM(
        output_dim=256,
        init='glorot_uniform',
        inner_init='orthogonal',
        forget_bias_init='one',
        W_regularizer=0.0001,
        return_sequences=False))
    model.add(Dropout(0.5))
    '''end'''


    # model.add(LSTM(
    #     output_dim=16,
    #     init='orthogonal',
    #     return_sequences=False
    # ))
    # model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=3,
        activation='linear'))
    # model.add(PReLU())
    # model.add(BatchNormalization())
    model.add(Activation("softmax"))
    optimizer = RMSprop(lr=learn_rate)
    #optimizer = SGD()

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy','categorical_accuracy','recall'])
    print "Compilation Time : ", time.time() - start
    return model

def build_model2():
    inp1 = Input(shape=(59,14,))
    inp2 = Input(shape=(1792,))
    out1 = LSTM(
        # input_dim=14,
        output_dim=128,
        init='glorot_uniform',
        inner_init = 'orthogonal',
        forget_bias_init='one',
        # W_regularizer=l2(0.0001),
        return_sequences=True)(inp1)
    out1 = GaussianDropout(0.5)(out1)
    out1_2 = BatchNormalization()(inp1)
    out1 = merge([out1, out1_2], mode='concat')
    out1 = LSTM(
        output_dim=256,
        init='glorot_uniform',
        inner_init='orthogonal',
        forget_bias_init='one',
        # W_regularizer=l2(0.0001),
        return_sequences=False)(out1)
    out1 = GaussianDropout(0.5)(out1)
    out1 = BatchNormalization()(out1)
    out2 = BatchNormalization()(inp2)
    # out2 = Dropout(0.8)(out2)

    # out2 = Dense(256, W_regularizer=l1(0.01), activity_regularizer=activity_l2(0.01))
    out = merge([out1, out2], mode='concat')
    # out = Dense(256, activation='linear', W_regularizer=l1(0.001))(out)
    out = BatchNormalization()(out)
    out = GaussianDropout(0.5)(out)
    out = Dense(512, activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dense(3, activation='softmax')(out)

    model = Model(input=[inp1, inp2], output=out)

    start = time.time()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy','categorical_accuracy','recall'])
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
                model.fit(x_train, y_train, batch_size=batch, nb_epoch=10, verbose=2, validation_split=0.1)
                score = model.evaluate(x_test, y_test, verbose=1, show_accuracy=True)
                params_result.append({'batch_size':batch, 'dropout_rate':dropout, 'score':score})
    return params_result
    # grid = GridSearchCV(estimator=model, param_grid=param_dict, verbose=1, n_jobs=-1)
    # grid_result = grid.fit(x_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # for params, mean_score, scores in grid_result.grid_scores_:
    #     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

def test_train(x_train, y_train, x_test, y_test, x_train_fft=None, x_test_fft=None):
    print 'length of y %s'%len(y_train)
    print 'dim of y %s'%len(y_train[0])
    # model = build_model(learn_rate=0.001)
    model = build_model2()
    # param_result = param_search(x_train, y_train, x_test=x_test, y_test=y_test)
    # print param_result
    # pickle.dump(open('params.pkl', 'wb'), param_result)
    # model.fit(x_train, y_train, batch_size=64, nb_epoch=10, verbose=2, validation_split=0.05)
    model.fit([x_train, x_train_fft], y_train, batch_size=64, nb_epoch=15, verbose=2, validation_split=0.05)
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

    score = model.evaluate([x_test, x_test_fft], y_test, verbose=2)

    print score

    #model.save('two_layer_lr0.001_60min.h5')

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data('hs_300_feature_60min.npy', 'hs_300_label_60min.npy')
    x_train_fft, x_test_fft = np.asarray(reshape_fft(x_train)), np.asarray(reshape_fft(x_test))
    x_train_fft, x_test_fft = normalize_2d(x_train_fft, x_test_fft)
    print 'fft dim is %s'%str(x_train_fft.shape)
    #model = build_model([14, 50, 100, 3])
    #model = KerasClassifier(build_fn=build_model, verbose=1, validation_split=0.1)

    test_train(x_train, y_train, x_test, y_test, x_train_fft, x_test_fft)