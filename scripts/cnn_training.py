#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import random
from sklearn import preprocessing
from keras.callbacks import History
import matplotlib.pyplot as plt
from keras.utils import np_utils
import os
from tensorflow.keras.optimizers import Adam
from plain_neural_network import*
from keras import backend as K


#---------------Data Preprocessing-------------#


#Importing ssh data

input_dir_ssh = "/albedo/home/ssunar/CNN_eddy_detection/test/months"
input_file_paths = sorted(
    [
        os.path.join(input_dir_ssh, fname)
        for fname in os.listdir(input_dir_ssh)
    ])
#If you want to exclude certain files

#input_file_paths.pop(29)
#input_file_paths.pop(11)

data_x = xr.open_mfdataset(input_file_paths,combine = 'nested', concat_dim="TIME")
data_x = data_x.ssh.to_numpy()
X = np.float32(data_x)
                            
X[X>1000] = 0 # Removing outliers or error values
X[X<-1000] = 0




#Importing segmentation data

"""
Segmentation mask
0 - background
1 - Cyclonic
2 - Antiyclonic
"""

input_dir_seg = "/albedo/home/ssunar/CNN_eddy_detection/test/segmentation_masks"
input_file_paths = sorted(
    [
        os.path.join(input_dir_seg, fname)
        for fname in os.listdir(input_dir_seg)
    ])
#If you want to exclude certain files

#input_file_paths.pop(29)
#input_file_paths.pop(11)

data_y = xr.open_mfdataset(input_file_paths,combine = 'nested', concat_dim="TIME")
data_y = data_y.seg_mask.to_numpy()
Y = np.float32(data_y)
Y[(Y != 1) & (Y!=2)] = 0 # Removing outliers or error values


#Taking 256x256 image sizes of the data to reduce the memory bias as we have data from same region.
#IMP!!!: The following split is done on the basis that our data dimension is 1200x480


temp_x_1 = X[:,0:256, 0:256]
temp_x_2 = X[:,0:256, 224:480]
temp_y_1 = Y[:,0:256, 0:256]
temp_y_2 = Y[:,0:256, 224:480]
data_x = np.concatenate((temp_x_1,temp_x_2), axis=0)
data_y = np.concatenate((temp_y_1,temp_y_2), axis=0)

for i in range(1,4):
    temp_x_1 = X[:,256*i:256*(i+1), 0:256]
    temp_x_2 = X[:,256*i:256*(i+1), 224:480]
    temp_y_1 = Y[:,256*i:256*(i+1), 0:256]
    temp_y_2 = Y[:,256*i:256*(i+1), 224:480]
    data_x = np.concatenate((data_x, np.concatenate((temp_x_1,temp_x_2), axis=0)), axis=0)
    data_y = np.concatenate((data_y, np.concatenate((temp_y_1,temp_y_2), axis=0)), axis=0)
    
print("Shape of data X:", data_x.shape)  
print("Shape of data Y:",data_y.shape)


#---------------Training-------------#


img_size = (256, 256)
num_classes = 3
batch_size = 16
epochs = 60
total_samples = len(data_x)
print(total_samples)

model = get_model(img_size, num_classes)
model.summary()

#Loss function 
#defined from the paper:
#Santana et al._2020_Neural network training for the detection and classification of oceanic mesoscale eddies

unique, counts = np.unique(data_y, return_counts=True)
dict(zip(unique, counts))

freq = [np.sum(counts)/j for j in counts]
weightsSeg = [f/np.sum(freq) for f in freq]
print(weightsSeg)

def dice_coef_anti(y_true, y_pred):
    smooth = 1.  # to avoid zero division
    y_true_anti = y_true[:,:,1]
    y_pred_anti = y_pred[:,:,1]
    intersection_anti = K.sum(y_true_anti * y_pred_anti)
    return (2 * intersection_anti + smooth) / (K.sum(y_true_anti)+ K.sum(y_pred_anti) + smooth)

def dice_coef_cyc(y_true, y_pred):
    smooth = 1.  # to avoid zero division
    y_true_cyc = y_true[:,:,2]
    y_pred_cyc = y_pred[:,:,2]
    intersection_cyc = K.sum(y_true_cyc * y_pred_cyc)
    return (2 * intersection_cyc + smooth) / (K.sum(y_true_cyc) + K.sum(y_pred_cyc) + smooth)

def dice_coef_nn(y_true, y_pred):
    smooth = 1.  # to avoid zero division
    y_true_nn = y_true[:,:,0]
    y_pred_nn = y_pred[:,:,0]
    intersection_nn = K.sum(y_true_nn * y_pred_nn)
    return (2 * intersection_nn + smooth) / (K.sum(y_true_nn) + K.sum(y_pred_nn) + smooth)
    
def mean_dice_coef(y_true, y_pred):
    return (dice_coef_anti(y_true, y_pred) + dice_coef_cyc(y_true, y_pred) + dice_coef_nn(y_true, y_pred))/3.

def weighted_mean_dice_coef(y_true, y_pred):
    #return (weightsSeg[2]*dice_coef_anti(y_true, y_pred) + weightsSeg[1]*dice_coef_cyc(y_true, y_pred) + weightsSeg[0]*dice_coef_nn(y_true, y_pred))
    return (weightsSeg[2]*dice_coef_anti(y_true, y_pred) + weightsSeg[1]*dice_coef_cyc(y_true, y_pred) + weightsSeg[0]*dice_coef_nn(y_true, y_pred))
      
def dice_coef_loss(y_true, y_pred):
    return 1 - weighted_mean_dice_coef(y_true, y_pred)


# Split our img set into a training and a validation set
split = 0.2
train_samples = int((1-split)*total_samples)
#same seed must be used
random.Random(0).shuffle(data_x)
random.Random(0).shuffle(data_y)
train_input = data_x[0:train_samples]
train_target = data_y[0:train_samples]
val_input = data_x[train_samples:total_samples]
val_target = data_y[train_samples:total_samples]

print("train_input:", train_input.shape)
print("val_input:", val_input.shape)

# Create data Sequences for each split
train_gen = plain_net_eddy(batch_size, img_size, train_input, train_target)
val_gen = plain_net_eddy(batch_size, img_size, val_input, val_target)
print("Size of each batch: ",train_gen[1][0].shape)



file_path_save = "/albedo/home/ssunar/CNN_eddy_detection/test/weights/weight" #This the name of file where the weights are saved
model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=['categorical_accuracy', mean_dice_coef, weighted_mean_dice_coef])

callbacks = [keras.callbacks.ModelCheckpoint(file_path_save, save_best_only=True , monitor='val_loss',save_weights_only=True, save_freq="epoch"),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=1e-30, min_lr=1e-30)]

history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks, shuffle=True,verbose=1)





