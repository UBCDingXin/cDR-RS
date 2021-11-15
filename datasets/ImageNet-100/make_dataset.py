#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 03:54:38 2019

@author: xin
"""


path_images = "./image/"
path_filename_to_label_plk = "./filename_to_classlabel.plk"

import os
import pickle
import h5py
import numpy as np
import gc
from tqdm import tqdm
from PIL import Image
import random
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

IMG_SIZE = 128
NC = 3
N_CLASS = 100
Resized_Method = "BILINEAR"; #"BILINEAR" or "LANCZOS"


np.random.seed(2019)
random.seed(2019)

#split into training and testing set
NValid = 10000
filenames_all = os.listdir(path_images) #all files under path_data directory
N_all = len(filenames_all)
NTrain = N_all - NValid
indx_valid = np.arange(N_all)
np.random.shuffle(indx_valid)
indx_valid = indx_valid[0:NValid]
indx_all = set(np.arange(N_all))
indx_train = np.array(list(indx_all.difference(indx_valid)))
assert set(indx_train).union(set(indx_valid)) == indx_all

filenames_train = []
for i in range(NTrain):
    filenames_train.append(filenames_all[indx_train[i]])

filenames_valid = []
for i in range(NValid):
    filenames_valid.append(filenames_all[indx_valid[i]])

with open(path_filename_to_label_plk,"rb") as f:
    filenamehead_to_classlabel = pickle.load(f)

# load images
images = np.zeros((N_all, NC, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
labels = np.zeros(N_all, dtype=np.int)

print("\n Begin loading image >>>")
for i in tqdm(range(N_all)):
    filename_cur = filenames_all[i]
    filename_head_cur = filename_cur.split('_')[0]
    label = filenamehead_to_classlabel[filename_head_cur]
    labels[i] = int(label)

    image = Image.open(path_images + filename_cur) #H,W,C
    if len(np.array(image).shape)==2: #gray scale image
        image = image.convert("RGB")
    image = np.array(image)
    image = image.transpose(2,0,1) #C,H,W

    #crop and resize images
    c, h, w = image.shape
    assert c==3
    short_side = h if h < w else w
    crop_size = short_side
    # Crop the center
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size

    image = image[:, top:bottom, left:right]
    _, h, w = image.shape
    image = image.transpose(1, 2, 0) #H,W,C
    if Resized_Method == "BILINEAR":
        image = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    elif Resized_Method == "LANCZOS": #best quality, worst performance
        image = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    image = np.array(image).transpose(2,0,1) #C,H,W
    images[i] = image.astype(np.uint8)
# end for


#show 10 example images every class; 10x100=1000
existing_classes = np.array([50, 137, 193, 25, 136, 255, 630, 256, 185, 263, 234, 179, 218, 42, 363, 92, 443, 349, 446, 447, 163, 451, 454, 461, 16, 192, 469, 470, 475, 486, 495, 960, 316, 501, 502, 7, 925, 530, 539, 995, 390, 547, 549, 952, 555, 575, 924, 589, 125, 54, 14, 606, 955, 374, 288, 33, 631, 326, 225, 645, 649, 676, 678, 683, 688, 704, 705, 706, 707, 723, 334, 397, 748, 82, 783, 150, 789, 798, 113, 804, 202, 28, 198, 831, 834, 89, 839, 76, 851, 853, 858, 873, 893, 346, 899, 41, 902, 909, 986, 340])
existing_classes = np.sort(existing_classes)
example_images = np.zeros((1000,NC,IMG_SIZE,IMG_SIZE),dtype=np.uint8)
assert set(labels) == set(existing_classes)
tmp = 0
for i in range(len(existing_classes)):
    indx_tmp = np.where(labels==existing_classes[i])[0]
    indx_tmp = indx_tmp[0:10]
    example_images[tmp:(tmp+10)] = images[indx_tmp]
    tmp+=10
example_images = example_images/255
example_images = torch.from_numpy(example_images).type(torch.float)
save_image(example_images, 'example_images.jpg', nrow=10)
del example_images; gc.collect()


######################################3
# dump to h5py file
images_train = images[indx_train]
labels_train = labels[indx_train]
images_valid = images[indx_valid]
labels_valid = labels[indx_valid]


### convert raw labels into 0,1,2,3,...
unique_labels_train = np.array(list(set(labels_train)))
unique_labels_train = np.sort(unique_labels_train)
assert len(unique_labels_train)==100
unique_labels_valid = np.array(list(set(labels_valid)))
unique_labels_valid = np.sort(unique_labels_valid)
assert len(unique_labels_train)==100
assert np.sum(unique_labels_train-unique_labels_valid)==0

label1000_to_label100 = dict()
label100_to_label1000 = dict()
for i in range(len(unique_labels_train)):
    label1000_to_label100[unique_labels_train[i]] = i
    label100_to_label1000[i] = unique_labels_train[i]

import pickle
with open('./label1000_to_label100.plk', 'wb') as handle:
    pickle.dump(label1000_to_label100, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./label100_to_label1000.plk', 'wb') as handle:
    pickle.dump(label100_to_label1000, handle, protocol=pickle.HIGHEST_PROTOCOL)

labels_train_new = np.zeros(labels_train.shape)
labels_valid_new = np.zeros(labels_valid.shape)
for i in range(len(labels_train)):
    labels_train_new[i] = label1000_to_label100[labels_train[i]]
for i in range(len(labels_valid)):
    labels_valid_new[i] = label1000_to_label100[labels_valid[i]]
labels_train_new = labels_train_new.astype(int)
labels_valid_new = labels_valid_new.astype(int)


with h5py.File("./ImageNet_128x128_100Class_processed.h5", "w") as f:
    f.create_dataset('images_train', data = images_train, dtype='uint8')
    f.create_dataset('labels_train_label1000', data = labels_train, dtype='int')
    f.create_dataset('labels_train', data = labels_train_new, dtype='int')
    f.create_dataset('images_valid', data = images_valid, dtype='uint8')
    f.create_dataset('labels_valid_label1000', data = labels_valid, dtype='int')
    f.create_dataset('labels_valid', data = labels_valid_new, dtype='int')
    f.create_dataset('existing_classes', data = existing_classes, dtype='int')
