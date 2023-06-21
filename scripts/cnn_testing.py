#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from keras.callbacks import History
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import Adam
from plain_neural_network import*
from sklearn.metrics import classification_report



img_size = (1200, 480)
num_classes = 3
input_dir_X = "/home/ollie/ssunar/ssh_filtered/months" #input dir for test ssh data
weight_path = "/home/ollie/ssunar/weights_filter_new/weights" #input dir for trained weights

preds_y_filtered = using_model(img_size, num_classes, input_dir_X, weight_path)
np.save('/home/ollie/ssunar/pred_data.npy', preds_y_filtered)

#Saves a 3d matrix of segmentaion mask where the third axis is time