#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:20:18 2022

@author: bivek
"""
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization, Activation, SeparableConv2D, Conv2DTranspose, Reshape, Dropout
import numpy as np
from keras.utils import np_utils
import os
import xarray as xr

def load_data_x(input_dir):
    input_file_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
        ])
    print(input_file_paths[0])
    data_x = xr.open_dataset(input_file_paths[0])
    data_x = data_x.ssh.to_numpy()
    data_x = np.float32(data_x)
    input_file_paths.pop(0)

    for abs_name in input_file_paths:
        print(abs_name)
        temp = xr.open_dataset(abs_name)
        temp = temp.ssh.to_numpy()
        temp = np.float32(temp)
        data_x = np.concatenate((data_x, temp), axis=0)
    return data_x

def load_data_y(input_dir):
    input_file_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
        ])
    print(input_file_paths[0])
    data_y = xr.open_dataset(input_file_paths[0])
    data_y = data_y.seg_mask.to_numpy()
    data_y = np.float32(data_y)
    input_file_paths.pop(0)
    
    for abs_name in input_file_paths:
        print(abs_name)
        temp = xr.open_dataset(abs_name)
        temp = temp.seg_mask.to_numpy()
        temp = np.float32(temp)
        data_y = np.concatenate((data_y, temp), axis=0)
    return data_y

class plain_net_eddy(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img, target_img):
        self.batch_size = batch_size
        self.img_size = img_size
        #self.data_x = xr.open_dataset(input_img)
        #self.data_y = xr.open_dataset(target_img)
        self.data_x = input_img
        self.data_y = target_img

    def __len__(self):
        return len(self.data_y) // self.batch_size
    

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img = self.data_x[i : i + self.batch_size]
        batch_target_img = self.data_y[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        #print(y.shape)
        for i in range(self.batch_size):
            x[i] = np.expand_dims(batch_input_img[i], 2)
            y[i] = np.expand_dims(batch_target_img[i], 2)
            
        y = np_utils.to_categorical(np.reshape(y[:,:,:,0],(self.batch_size,self.img_size[0]*self.img_size[1])),3)
        return x, y

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))
    #filters = 16
    
    for_concat = []
    p = inputs
    for index, dropout_filter in zip(range(1,5,1),[(0.2,16),(0.3,16),(0.4,32),(0.5,32)]): 
        x = layers.SeparableConv2D(dropout_filter[1], 3, padding="same", use_bias=False)(p)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.SeparableConv2D(dropout_filter[1], 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = Dropout(dropout_filter[0])(x)
        for_concat.append(x)
        
        if index<4:
            p = layers.MaxPooling2D(pool_size=(2, 2))(x)
        elif index==4:
            p = x

    for index, dropout_filter in zip(range(3,0,-1),[(0.4,32),(0.3,16),(0.2,16)]):
        x = concatenate([UpSampling2D((2,2))(p), for_concat[index-1]])
        x = SeparableConv2D(dropout_filter[1], 3, padding="same", use_bias=False)(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = SeparableConv2D(dropout_filter[1], 3, padding="same", use_bias=False)(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_filter[0])(x)
        p = x

    # Add a per-pixel classification layer
    
    X = SeparableConv2D(num_classes, (1,1), padding="same", use_bias=False)(x)   
    X = Reshape((img_size[0] * img_size[1], num_classes))(X) 
    outputs = Activation("softmax")(X)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def using_model(img_size, num_classes, input_dir_X, weight_path):
    input_file_paths = sorted(
    [
        os.path.join(input_dir_X, fname)
        for fname in os.listdir(input_dir_X)
    ])
    data_X = xr.open_mfdataset(input_file_paths,combine = 'nested', concat_dim="TIME")
    data_X = data_X.ssh.to_numpy()
    data_X = np.float32(data_X)

    data_X[data_X>1000] = 0
    pred_x = np.reshape(data_X,(len(data_X),img_size[0],img_size[1],1))
    model = get_model(img_size, num_classes)
    model.load_weights(weight_path)
    preds_y = model.predict(pred_x)
    mask = np.argmax(np.reshape(preds_y,(len(data_X),img_size[1],img_size[0],num_classes)), axis=-1)
    preds_y = np.reshape(mask,(len(data_X), img_size[0],img_size[1]))
    return preds_y

# Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()

# Build model
#model = get_model(img_size, num_classes)
#model.summary()

