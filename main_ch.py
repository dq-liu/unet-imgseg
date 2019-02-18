# !/usr/bin/python3

### Description    : Model training and evaluation; predictions. RGB CHANNELS SEPERATED.
### Version        : Python = 3.5; Tensorflow = 1.8.
### Author         : ql82.
### Created        : 2018/09/30
### Last updated   : 2019/02/17


import os
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

import warnings
warnings.filterwarnings('ignore')
import pickle, ast, sys, random, math, sklearn, gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import roc_auc_score
from skimage.exposure import equalize_hist
from skimage.transform import resize
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.morphology import label
from keras.preprocessing.image import ImageDataGenerator
from itertools import chain
from keras.models import Model
from keras import losses
from keras.layers import Input, merge, Lambda, Softmax, Convolution2D, Convolution3D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import roc_auc_score
from model_ch import *
from obj import *


# read in images, masks and labels
images = pickle.load(open('images_rec.p', 'rb'))
masks = pickle.load(open('masks_rec.p', 'rb'))
labels = pickle.load(open('labels_rec.p', 'rb'))

# channel seperation and hist equalization
images_r = images[:, :, :, 0].reshape((90, 966, 1296, 1))
images_g = images[:, :, :, 1].reshape((90, 966, 1296, 1))
images_b = images[:, :, :, 2].reshape((90, 966, 1296, 1))

images_eq_r = np.array([equalize_hist(images_r[i], mask=labels[i]) for i in range(90)])
images_eq_g = np.array([equalize_hist(images_g[i], mask=labels[i]) for i in range(90)])
images_eq_b = np.array([equalize_hist(images_b[i], mask=labels[i]) for i in range(90)])

# split data into training and test
random.seed = 2019
train_ix = random.sample(range(90), 80)
test_ix = np.delete(list(range(90)), train_ix)

images_r_train = images_eq_r[train_ix].astype('float32')
images_g_train = images_eq_g[train_ix].astype('float32')
images_b_train = images_eq_b[train_ix].astype('float32')
labels_train = labels[train_ix].astype('float32')

images_r_test = images_eq_r[test_ix].astype('float32')
images_g_test = images_eq_g[test_ix].astype('float32')
images_b_test = images_eq_b[test_ix].astype('float32')
labels_test = labels[test_ix].astype('float32')

# compile model for r channel
tf.reset_default_graph()
model_r = get_unet(loss_fun=crossentropy_w, metric_list=[auc])
model_checkpoint_r = ModelCheckpoint('unet_r.hdf5', monitor='loss', verbose=1, save_best_only=True)
model_r.fit(images_r_train, labels_train, validation_split=0.1, batch_size=4, nb_epoch=40, verbose=1, shuffle=True, callbacks=[model_checkpoint_r])

# model for g channel
tf.reset_default_graph()
model_g = get_unet(loss_fun=crossentropy_w, metric_list=[auc])
model_checkpoint_g = ModelCheckpoint('unet_g.hdf5', monitor='loss', verbose=1, save_best_only=True)
model_g.fit(images_g_train, labels_train, validation_split=0.1, batch_size=4, nb_epoch=40, verbose=1, shuffle=True, callbacks=[model_checkpoint_g])

# model for b channel
tf.reset_default_graph()
model_b = get_unet(loss_fun=crossentropy_w, metric_list=[auc])
model_checkpoint_b = ModelCheckpoint('unet_b.hdf5', monitor='loss', verbose=1, save_best_only=True)
model_b.fit(images_b_train, labels_train, validation_split=0.1, batch_size=4, nb_epoch=40, verbose=1, shuffle=True, callbacks=[model_checkpoint_b])
