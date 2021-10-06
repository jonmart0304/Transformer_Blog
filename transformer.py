'''
///-----------------------------------------------------------------
///	File Name:			transformer.py
///	Description:		Custom transformer code (https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3)
///	Author:				Jonathan Martinez
///	Date:				10/6/2021
///	Notes:				Please read the following instructions to run the code
///	Contact:			Embedded Signal Processing Lab, Texas A&M University
///	Revision History:
///	Name: Rev 00			Date: 10/6/2021			Description: Baseline model for Bio-Z
///-----------------------------------------------------------------
'''

import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt
import random
import os
import sys
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import defaultdict
import time
from tensorflow.keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import shap
import json
import keras
from keras_self_attention import SeqSelfAttention
from keras.callbacks import History
from keras.preprocessing.sequence import pad_sequences
import glob
#First let's import
import heartpy as hp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.core.pylabtools import figsize
from sklearn import preprocessing
import biosppy
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import butter, lfilter, freqz
import neurokit2 as nk
from scipy.signal import butter, filtfilt, resample
import wfdb
from os import path
import sklearn
import math

class Time2Vec(keras.layers.Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))
    
from tensorflow_addons.layers import MultiHeadAttention

class AttentionBlock(keras.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1) 

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x
    
class ModelTrunk(keras.Model):
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

        
    def call(self, inputs):
        time_embedding = keras.layers.TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        return K.reshape(x, (-1, x.shape[1] * x.shape[2])) # flat vector of features out

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


data_directory = '/data/datasets/bioz/b2b/'
subjects = next(os.walk("/data/datasets/bioz/b2b/"))[1]
data_dict = defaultdict(dict)
for subject in subjects:
    bioz_lists = pd.read_pickle(data_directory + subject + "/list_data_time_100.pckl")
    bioz_dtw_lists = np.load("/home/grads/j/jmartinez0304/Research/Transformers/AttendAndDiagnose/playground/DTW_Data/OneForAll_Sub3/PtsMask/" + subject + "/list_data_time_100_dtwGuidedPts.npy", allow_pickle=True)
    new_bioz_list,new_bioz_dtw_list = list(),list()
    for raw_b,d in zip(bioz_lists,bioz_dtw_lists):
      b = np.gradient(raw_b,axis=0)
      #b = raw_b
      tmp_zeros = np.zeros((100-len(b),5))
      pad_b = np.concatenate((b,tmp_zeros),axis=0)
      new_bioz_list.append(pad_b)
      tmp_dtw_zeros = np.zeros((100-len(d),4))
      pad_d = np.concatenate((d,tmp_dtw_zeros),axis=0)
      new_bioz_dtw_list.append(pad_d)
    bioz_lists = np.asarray(new_bioz_list)
    bioz_dtw_lists = np.asarray(new_bioz_dtw_list)
    bioz_labels = pd.read_pickle(data_directory + subject + "/labels_100.pckl")
    bioz_labels = bioz_labels.values
    data_dict[subject]['X_data'] = bioz_lists[:,:,0:2]
    data_dict[subject]['X_dtw_data'] = bioz_dtw_lists[:,:,0:2]
    data_dict[subject]['BP_data'] = bioz_labels
    
    
testSubject = 'Subject7'
for subject in subjects:
    if subject != testSubject:
        continue
    print(subject)
    print('Loading data...')
    
    save_directory = "Results/transformer/personalized/dbp/e50/"
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    X_data = np.asarray(data_dict[subject]['X_data'])
    X_dtw_data = np.asarray(data_dict[subject]['X_dtw_data'])
    sbp_data = np.asarray(data_dict[subject]['BP_data'])[:,1]
    dbp_data = np.asarray(data_dict[subject]['BP_data'])[:,0]
    indices = np.arange(len(X_data))
    np.random.shuffle(indices)
    X_data = X_data[indices]
    X_dtw_data = X_dtw_data[indices]
    sbp_data = sbp_data[indices]
    dbp_data = dbp_data[indices]
    bp_data = np.concatenate((np.reshape(dbp_data,(len(dbp_data),1)),np.reshape(sbp_data,(len(sbp_data),1))),axis=1)
    
    training_size = 0.8
    x_train = X_data[0:int(len(X_data)*training_size)]     
    x_dtw_train = X_dtw_data[0:int(len(X_dtw_data)*training_size)]
    sbp_train = sbp_data[0:int(len(sbp_data)*training_size)]
    dbp_train = dbp_data[0:int(len(dbp_data)*training_size)]
    bp_train = bp_data[0:int(len(bp_data)*training_size)]
    
    x_test = X_data[int(len(X_data)*training_size):]     
    x_dtw_test = X_dtw_data[int(len(X_dtw_data)*training_size):]
    sbp_test = sbp_data[int(len(sbp_data)*training_size):]
    dbp_test = dbp_data[int(len(dbp_data)*training_size):]
    bp_test = bp_data[int(len(bp_data)*training_size):]


    X_train = x_train
    X_test = x_test
    X_dtw_train = x_dtw_train
    X_dtw_test = x_dtw_test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Channel 1 inputs
    ch1_X_train = X_train[:,:,0]
    ch1_X_test = X_test[:,:,0]
    ch1_X_dtw_train = x_dtw_train[:,:,0]
    ch1_X_dtw_test = x_dtw_test[:,:,0]


    # Channel 2 inputs
    ch2_X_train = X_train[:,:,1]
    ch2_X_test = X_test[:,:,1]
    ch2_X_dtw_train = x_dtw_train[:,:,1]
    ch2_X_dtw_test = x_dtw_test[:,:,1]
    
    ch1_X_train = np.reshape(ch1_X_train,(len(ch1_X_train),len(ch1_X_train[0]),1))
    ch2_X_train = np.reshape(ch2_X_train,(len(ch2_X_train),len(ch2_X_train[0]),1))
    ch1_X_test = np.reshape(ch1_X_test,(len(ch1_X_test),len(ch1_X_test[0]),1))
    ch2_X_test = np.reshape(ch2_X_test,(len(ch2_X_test),len(ch2_X_test[0]),1))

    if not os.path.exists(save_directory + subject + "/models/"):
        os.makedirs(save_directory + subject + "/models/")

    print('Init model...')
    

    wave_in = tf.keras.Input(shape=X_train[0].shape)
    
    wave_in = tf.keras.Input(shape=X_train[0].shape)
    tr1_h1 = ModelTrunk(num_heads=4, head_size=64, ff_dim=None, num_layers=1)(wave_in)
    f1_h1 = tf.keras.layers.Flatten()(tr1_h1)
    #dbp_out = tf.keras.layers.Dense(1)(f1_h1)
    bp_out = tf.keras.layers.Dense(1)(f1_h1)
    
    model = keras.Model(
        inputs=[wave_in],
        outputs=[bp_out],
    )
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(
        optimizer=opt,
        loss=[
            'mse'
        ]
    )
    
    model.summary()

    history = History()
    # checkpoint
    if not os.path.exists(save_directory + subject + '/models/'):
        os.makedirs(save_directory + subject + '/models/')
    weights_filepath = save_directory + subject + '/models/best_weights_exp1.hdf5'
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    callbacks_list = [history]

    model.fit(X_train, dbp_train, validation_split=0.1, shuffle=True, epochs=100, batch_size=16, callbacks=callbacks_list)

    model.save_weights(weights_filepath)
    #model.load_weights(weights_filepath)

    preds = model.predict(X_test)

    np.save(save_directory + subject + '/X_test',X_test)
    np.save(save_directory + subject + '/BP_test',bp_test)
    np.save(save_directory + subject + '/preds',preds)
    
    K.clear_session()
    
    print('Done!')  
    
save_directory = "Results/transformer/personalized/dbp/e50/"
    
BP_test = np.load(save_directory + testSubject + '/BP_test.npy',allow_pickle=True)
preds = np.load(save_directory + testSubject + '/preds.npy',allow_pickle=True)
    
print('RMSE... ' + str(math.sqrt(sklearn.metrics.mean_squared_error(BP_test[:,0], preds[:,0]))))
print('Correlation... ' + str(pearsonr(np.asarray(BP_test[:,0]), np.asarray(preds[:,0]))))

plt.figure()
plt.scatter(BP_test[:,0],preds[:,0])
plt.show()
