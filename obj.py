import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from itertools import chain
from keras.models import Model
from keras import losses
from keras.layers import Input, merge, Lambda, Softmax, Convolution2D, Convolution3D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K


def crossentropy_w(y_true, y_pred, pos_weight=3.5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    logloss = K.mean(y_true_f*-K.log(y_pred_f)*pos_weight + (1-y_true_f)*-K.log(1-y_pred_f), axis=-1)
    return logloss

def crossentropy(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
    logloss = K.mean(K.binary_crossentropy(y_true, y_pred))
    return logloss

## eval metrics
# def dice_coef(y_true, y_pred, smooth=1):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.dot(y_true, K.transpose(y_pred))
#     union = K.dot(y_true, K.transpose(y_true)) + K.dot(y_pred, K.transpose(y_pred))
#     return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (union + smooth))

def auc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return tf.py_func(roc_auc_score, (y_true_f, y_pred_f), tf.double)

# def auc(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     return roc_auc_score(y_true_f, y_pred_f)

# def dice_coef(y_true, y_pred, smooth=0.):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)