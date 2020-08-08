#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 23:28:48 2018

@author: Rohit Pardasani & Navchetan Awasthi
"""


# Creating and training model

import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras import backend as Ks
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras.callbacks import ModelCheckpoint,CSVLogger

def customLoss(yTrue,yPred):
    return (1e4)*(Ks.mean(Ks.square(yPred - yTrue), axis=-1))

cload = 'B'
csave = 'C'
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 100
N_TEST_SAMPLES = 13

CHANNELS = 1
BATCH_SIZE = 100
LEARNING_RATE = 2e-8
N_LAYERS = 7
F = 64
 

folderNameInput = './TestData/'
folderSaveOutput = './TestData/'
X_TEST = np.zeros((N_TEST_SAMPLES,IMAGE_HEIGHT,IMAGE_WIDTH))

print('Loading Testing Data')
for i in range(N_TEST_SAMPLES):
    if(i%1000 == 0):
        print(i)
    pathr = folderNameInput+cload+str(i+1)+'.mat'
    x = sio.loadmat(pathr)
    X_TEST[i,:,:] = x['Rref']


if(np.isnan(X_TEST).any()==True):
    print('There is nan value in X_TEST')


X_TEST = X_TEST.reshape(X_TEST.shape[0],IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS)


myModel = Sequential()

firstLayer = Convolution2D(F, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS), use_bias=True, bias_initializer='zeros')
myModel.add(firstLayer)
myModel.add(Activation('relu'))

for i in range(N_LAYERS-2):
    Clayer = Convolution2D(F, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,F), use_bias=True, bias_initializer='zeros')
    myModel.add(Clayer)
    Blayer = BatchNormalization(axis=-1, epsilon=1e-3)
    myModel.add(Blayer)
    myModel.add(Activation('relu'))
    
lastLayer = Convolution2D(CHANNELS, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,F), use_bias=True, bias_initializer='zeros')
myModel.add(lastLayer)    



# this is to get weights of previously saved model
savedModel = keras.models.load_model('modelSaved1.h5', custom_objects={'customLoss': customLoss})

myadam = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

myModel.set_weights(savedModel.get_weights())
myModel.compile(loss=customLoss, optimizer=myadam, metrics=['accuracy'])
myModel.summary()

Z_TEST = myModel.predict(X_TEST,batch_size=1, verbose=1)
print('output predicted')
Y_TEST = X_TEST - Z_TEST
for i in range(N_TEST_SAMPLES):
    patht = folderSaveOutput+csave+str(i+1)+'.mat'
    Rref = Y_TEST[i,:,:,:]
    sio.savemat(patht,{'Rref':Rref})


