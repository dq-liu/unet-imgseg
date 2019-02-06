# !/usr/bin/python3

### Description    : Defines the Unet framework.
### Version        : Python = 3.5; Tensorflow = 1.8.
### Author         : ql82.
### Created        : 2018/09/30
### Last updated   : 2019/01/30


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from itertools import chain
from keras.models import Model
from keras import losses
from keras.layers import Input, merge, Lambda, Softmax, Convolution2D, Convolution3D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

def get_unet(loss_fun, metric_list):
	'''The U-net framework: input image dim = (966, 1296, 3)'''

    K.clear_session() 
    
    inputs = Input((966, 1296, 3))                                             # 966, 1296, 3
    conv0 = Convolution2D(1, 1, activation='relu', padding='same')(inputs)     # 966, 1296, 1
     
    conv1 = Convolution2D(32, 3, activation='relu', padding='same')(conv0)     # 966, 1296, 32
    conv1 = Convolution2D(32, 3, activation='relu', padding='same')(conv1)     # 966, 1296, 32 --------
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                              # 483, 648, 32
     
    conv2 = Convolution2D(64, 3, activation='relu', padding='same')(pool1)     # 483, 648, 64
    conv2 = Convolution2D(64, 3, activation='relu', padding='same')(conv2)     # 483, 648, 64  ------
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                              # 241, 324, 64
     
    conv3 = Convolution2D(128, 3, activation='relu', padding='same')(pool2)    # 241, 324, 128
    conv3 = Convolution2D(128, 3, activation='relu', padding='same')(conv3)    # 241, 324, 128 ----
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)                              # 120, 162, 128
    
    conv4 = Convolution2D(256, 3, activation='relu', padding='same')(pool3)    # 120, 162, 256
    conv4 = Convolution2D(256, 3, activation='relu', padding='same')(conv4)    # 120, 162, 256 --
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)                              # 60, 81, 256
    
    conv5 = Convolution2D(512, 3, activation='relu', padding='same')(pool4)    # 60, 81, 512
    conv5 = Convolution2D(512, 3, activation='relu', padding='same')(conv5)    # 60, 81, 512
    
    up6 = UpSampling2D(size=(2, 2))(conv5)                                     # 120, 162, 512
    up6 = Convolution2D(256, 2, activation='relu', padding='same')(up6)        # 120, 162, 256 --
    up6 = merge.concatenate([up6, conv4])                                      # 120, 162, 512
    conv6 = Convolution2D(256, 3, activation='relu', padding='same')(up6)      # 120, 162, 256
    conv6 = Convolution2D(256, 3, activation='relu', padding='same')(conv6)    # 120, 162, 256
    conv6 = ZeroPadding2D((1, 0))(conv6)                                       # 122, 162, 256
    
    up7 = UpSampling2D(size=(2, 2))(conv6)                                     # 244, 324, 256
    up7 = Convolution2D(128, 2, activation='relu', padding='same')(up7)        # 244, 324, 128 ----
    up7 = merge.concatenate([Lambda(lambda x: x[:, 1:242])(up7), conv3])       # 241, 324, 256
    conv7 = Convolution2D(128, 3, activation='relu', padding='same')(up7)      # 241, 324, 128
    conv7 = Convolution2D(128, 3, activation='relu', padding='same')(conv7)    # 241, 324, 128
    conv7 = ZeroPadding2D((1, 0))(conv7)                                       # 243, 324, 128
    
    up8 = UpSampling2D(size=(2, 2))(conv7)                                     # 486, 648, 128
    up8 = Convolution2D(64, 2, activation='relu', padding='same')(up8)         # 486, 648, 64  ------
    up8 = merge.concatenate([Lambda(lambda x: x[:, 1:484])(up8), conv2])       # 483, 648, 128
    conv8 = Convolution2D(64, 3, activation='relu', padding='same')(up8)       # 483, 648, 64
    conv8 = Convolution2D(64, 3, activation='relu', padding='same')(conv8)     # 483, 648, 64
    
    up9 = UpSampling2D(size=(2, 2))(conv8)                                     # 966, 1296, 64
    up9 = Convolution2D(32, 2, activation='relu', padding='same')(up9)         # 966, 1296, 32 --------
    up9 = merge.concatenate([up9, conv1])                                      # 966, 1296, 64
    conv9 = Convolution2D(32, 3, activation='relu', padding='same')(up9)       # 966, 1296, 32
    conv9 = Convolution2D(32, 3, activation='relu', padding='same')(conv9)     # 966, 1296, 32
    
    conv10 = Convolution2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=loss_fun, metrics=metric_list)

    return model