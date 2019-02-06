# !/usr/bin/python3

### Description    : Data augmentation; label generation.
### Version        : Python = 3.5; Tensorflow = 1.8.
### Author         : ql82.
### Created        : 2018/09/30
### Last updated   : 2019/01/30


import pickle, ast, sys, random, math, sklearn, gc
import numpy as np
import pandas as pd
from glob import glob
from skimage.transform import resize
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.morphology import label


# read in images and masks
images = pickle.load(open('images_rec.p', 'rb'))
masks = pickle.load(open('masks_rec.p', 'rb'))

# create labels by setting ranges
labels = []
ones = []
for i in range(len(images)):
    print(i)
    x = images[i]
    y = masks[i]
    label = np.empty((966, 1296, 1))
    for j in range(966):
        for k in range(1296):
            if np.sqrt((y[j][k][0]-255)**2+(y[j][k][1]-255)**2+(y[j][k][2]-0)**2)<=100:
                label[j][k] = 1
            else:
                label[j][k] = 0
    labels.append(label)
    ones.append(np.array(label).sum()/(966*1296))
labels = np.array(labels).astype(int)

# check the percentage of labeled pixels
pd.DataFrame(ones).describe(percentiles=[.25, .5, .75, .95, .99]).T

# save labels to a pickle file
pickle.dump(labels, open('labels_rec.p', 'wb'))