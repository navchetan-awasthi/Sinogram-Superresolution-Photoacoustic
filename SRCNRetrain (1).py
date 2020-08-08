#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 23:28:48 2018

@author: Rohit Pardasani & Navchetan Awasthi
"""

# Creating and training model
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"



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


IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
N_TRAIN_SAMPLES = 200000
CHANNELS = 1
N_EVALUATE_SAMPLES = 30000
EVALUATE_FROM = 200001
BATCH_SIZE = 100
EPOCHS = 10000
LEARNING_RATE = 2e-8
N_LAYERS = 7
F = 64


folderNameInput = './SinoPatches/'
folderNameDiff = './SinoPatches/'
X_TRAIN = np.zeros((N_TRAIN_SAMPLES,IMAGE_HEIGHT,IMAGE_WIDTH))
Y_TRAIN = np.zeros((N_TRAIN_SAMPLES,IMAGE_HEIGHT,IMAGE_WIDTH))
X_EVALUATE = np.zeros((N_EVALUATE_SAMPLES,IMAGE_HEIGHT,IMAGE_WIDTH))
Y_EVALUATE = np.zeros((N_EVALUATE_SAMPLES,IMAGE_HEIGHT,IMAGE_WIDTH))

print('Loading Training Data')
for i in range(N_TRAIN_SAMPLES):
    if(i%1000 == 0):
        print(i)
    pathr = folderNameInput+'P'+str(i+1)+'.mat'
    x = sio.loadmat(pathr)
    X_TRAIN[i,:,:] = x['P']
    patht = folderNameDiff+'D'+str(i+1)+'.mat'
    y = sio.loadmat(patht)
    Y_TRAIN[i,:,:] = y['D']


print('Loading Validation Data')
for i in range(N_EVALUATE_SAMPLES):
    if(i%1000 == 0):
        print(i)
    pathr = folderNameInput+'P'+str(EVALUATE_FROM+i)+'.mat'
    x = sio.loadmat(pathr)
    X_EVALUATE[i,:,:] = x['P']
    patht = folderNameDiff+'D'+str(EVALUATE_FROM+i)+'.mat'
    y = sio.loadmat(patht)
    Y_EVALUATE[i,:,:] = y['D']

if(np.isnan(X_TRAIN).any()==True):
    print('There is nan value in X_TRAIN')

if(np.isnan(Y_TRAIN).any()==True):
    print('There is nan value in Y_TRAIN')

if(np.isnan(X_EVALUATE).any()==True):
    print('There is nan value in X_EVALUATE')

if(np.isnan(Y_EVALUATE).any()==True):
    print('There is nan value in Y_EVALUATE')



X_TRAIN = X_TRAIN.reshape(X_TRAIN.shape[0],IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS)
Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0],IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS)      
X_EVALUATE = X_EVALUATE.reshape(X_EVALUATE.shape[0],IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS)
Y_EVALUATE = Y_EVALUATE.reshape(Y_EVALUATE.shape[0],IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS)   

datagentrain = ImageDataGenerator()
datagenvalid = ImageDataGenerator()

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

csv_logger = CSVLogger('trainingLog2.log')
model_checkpoint = ModelCheckpoint('modelSaved2.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
myModel.fit_generator(datagentrain.flow(X_TRAIN, Y_TRAIN, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(X_TRAIN)/BATCH_SIZE, validation_data=datagenvalid.flow(X_EVALUATE, Y_EVALUATE, batch_size=BATCH_SIZE),validation_steps=len(X_EVALUATE)/BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[csv_logger,model_checkpoint])
score = myModel.evaluate(X_EVALUATE, Y_EVALUATE, batch_size=BATCH_SIZE)




