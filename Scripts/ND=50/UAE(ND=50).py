#!/usr/bin/env python
# coding: utf-8


import struct
import time
import os
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sklearn
from sklearn.preprocessing import LabelBinarizer
# model training and test
from contextlib import redirect_stdout
# from tensorflow import keras
from keras import layers
import math
import tensorflow as tf

import keras
from keras.models import Model, Sequential
from keras.layers import core, Flatten, Dense, Input, Dropout, Activation, Conv1D, MaxPooling1D, Reshape, AveragePooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import to_categorical
from keras.models import load_model


### Scripts based on gaibzai github: https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA


def load_data(file, load_metadata=False):
    #check_file_exists(file)
    # Open the database HDF5 for reading
    try:
        in_file  = h5py.File(file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    
    print(X_profiling.shape[0])
    print(X_profiling.shape[1])
    
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], 
                                                                  in_file['Attack_traces/metadata'])



labeled_file='./desync_50/0data/ASCAD_desync50_Sbox3_3000_test.h5'
(X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_data(labeled_file, load_metadata=True)
print(X_profiling.shape)
print(X_attack.shape)


Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)),X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))


# define AE model

def ae_cnn_model(encoding_dim = 60,kernal=4,kernal_size=100):
    input_shape = (3000,1)
    encoder_input = x = keras.Input(shape=input_shape, name='encoder_input')
    # first cov block
    x = layers.Conv1D(kernal, kernal_size, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)   
    x = layers.AveragePooling1D(50, strides=50, name='block2_pool1')(x)
    #x = layers.Conv1D(4, 5, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv2')(x)   
    #x = layers.AveragePooling1D(5, strides=5, name='block2_pool2')(x)
    x = layers.Flatten(name='flatten')(x)
    node = x._keras_shape[1] #int(240*(100/kernal_size)*(kernal/4))
    encoder_output = layers.Dense(encoding_dim)(x)
    encoder = keras.Model(encoder_input, encoder_output, name='encoder')
    encoder.summary()
    
    decoder_input = x = keras.Input(shape=(encoding_dim, ), name='decoder_input') 
    x = layers.Dense(node, activation='relu')(x)
    x = layers.Dense(3000)(x)
    decoder_output = Reshape((3000,1))(x)
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    decoder.summary()
    autoencoder_input = keras.Input(shape=input_shape, name='autoencoder_input')
    encoded = encoder(autoencoder_input)
    autoencoder_output = decoder(encoded)
    autoencoder = keras.Model(
        autoencoder_input,
        autoencoder_output,
        name='autoencoder',
    )
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    modelSummary = './asymetric_ASCAD_desync50_AE(CNN)_'+str(kernal)+'x'+str(kernal_size)+'layers_model_summary.txt'
    print(modelSummary)
    with open(modelSummary, 'w') as f:
        with redirect_stdout(f):
            encoder.summary()
            decoder.summary()
            autoencoder.summary()
    return encoder,autoencoder,decoder


file_path = 'desync_50/data' # *********** change No. here
if (os.path.exists(file_path)==False):
    os.makedirs(file_path)

# **************change AE model hyperparameter*******************

dims = [75]
kernal_number = [8]
kernal_sizes = [100]
for encoding_dim in dims:
    for kernal in kernal_number:
        for kernal_size in kernal_sizes:
            start = time.time() # time
            model_weights = file_path + '/ae_model/ASCAD_desync50_AE(CNN)_'+str(encoding_dim)+'dim_'+str(kernal)+'x'+str(kernal_size)+'layers_model_weights.h5'  
            callbacks_list = [
                keras.callbacks.ModelCheckpoint(
                    filepath=model_weights, 
                    monitor='val_loss',
                    save_best_only=True,
                )
            ]
            encoder,autoencoder,decoder = ae_cnn_model(encoding_dim,kernal,kernal_size)
            history=autoencoder.fit(Reshaped_X_profiling, Reshaped_X_profiling,
                            epochs=40,
                            batch_size=512,
                            shuffle=False,
                            validation_data=(Reshaped_X_test, Reshaped_X_test),
                            callbacks=callbacks_list)
            end=time.time() # time
            print('Temps execution = %d'%(end-start))

            lossy = history.history['loss']
            val_lossy = history.history['val_loss']
            np_lossy =np.array(lossy).reshape((1,len(lossy))) 
            np_val_lossy =np.array(val_lossy).reshape((1,len(val_lossy))) 

            np_out = np.concatenate([np_lossy,np_val_lossy],axis=0)
            #loss_file = file_path +'/loss_ASCAD_AE(MLP)_'+str(node)+'node_'+str(encoding_dim)+'encoding_dim_'+str(layer_nb)+'layers.txt'
            #np.savetxt(loss_file,np_out)    
            print("保存文件成功")
            #divide trace set
            subfile_name = file_path+'_res/ASCAD_desync50_Sbox3_3000_test_'+str(encoding_dim)+'_AE(kernal_'+str(kernal)+'x'+str(kernal_size)+').h5'

            labels_profiling = []
            labels_attack = []


            raw_traces_profiling = encoder.predict(Reshaped_X_profiling)
            raw_traces_attack = encoder.predict(Reshaped_X_test)
            labels_profiling = Y_profiling
            labels_attack = Y_attack


            with h5py.File(subfile_name,'w') as f:
                #f.create_dataset('test_numpy',data=x)
                profiling_traces_group = f.create_group('Profiling_traces')
                attack_traces_group = f.create_group("Attack_traces")
                # Datasets in the groups
                profiling_traces_group.create_dataset(name="traces", data=raw_traces_profiling)
                attack_traces_group.create_dataset(name="traces", data=raw_traces_attack)
                # Labels in the groups
                profiling_traces_group.create_dataset(name="labels", data=labels_profiling)
                attack_traces_group.create_dataset(name="labels", data=labels_attack)

                # meta data in the groups
                profiling_traces_group.create_dataset(name="metadata",data = Metadata_profiling)
                attack_traces_group.create_dataset(name="metadata", data=Metadata_attack)

