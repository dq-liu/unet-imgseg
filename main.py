# !/usr/bin/python3

### Description    : Model training and evaluation; predictions.
### Version        : Python = 3.5; Tensorflow = 1.8.
### Author         : ql82.
### Created        : 2018/09/30
### Last updated   : 2019/01/30


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
from model import *
from obj import *


# read in images, masks and labels
images = pickle.load(open('images_rec.p', 'rb'))
masks = pickle.load(open('masks_rec.p', 'rb'))
labels = pickle.load(open('labels_rec.p', 'rb'))

# split data into training and test
import random
random.seed = 2019
train_ix = random.sample(range(90), 80)
test_ix = np.delete(list(range(90)), train_ix)
images_train = images[train_ix].astype('float32')
labels_train = labels[train_ix].astype('float32')
images_test = images[test_ix].astype('float32')
labels_test = labels[test_ix].astype('float32')

# compile cnn model
tf.reset_default_graph()
model = get_unet(loss_fun=crossentropy_w, metric_list=[auc])

# fit model
checkPoint_ = 'unet0201.hdf5'
validation_split_ = .1
batch_size_ = 4
nb_epoch_ = 60
model_checkpoint_ = ModelCheckpoint(checkPoint_, monitor='loss', verbose=1, save_best_only=True)
model.fit(images_train, labels_train, validation_split=validation_split_, batch_size=batch_size_, nb_epoch=nb_epoch_, verbose=1, shuffle=True, callbacks=[model_checkpoint_])

# predict on test set
image_pred = model.predict(images_test)

# check the number of predicted labeled pixels and ground truth
oneInLabel = [labels[i].sum() for i in range(10)]
oneInPred = [(image_pred[i]>0.4).sum() for i in range(10)]
pd.DataFrame({'Label':oneInLabel, 'Pred':oneInPred})

# visulize predicted labels
test_pred = image_pred[9]
test_pred_plot = np.empty((966, 1296, 3))
for i in range(966):
    for j in range(1296):
        if test_pred[i][j] > 0.40:
            test_pred_plot[i][j] = [255, 255, 255]
        else:
            test_pred_plot[i][j] = [0, 0, 0]
imshow(test_pred_plot.astype(np.uint8))