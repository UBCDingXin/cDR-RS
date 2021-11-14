import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
import sys
import PIL
from PIL import Image
from tqdm import trange, tqdm

### import my stuffs ###
from models import *


# ################################################################################
# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')


################################################################################
# torch dataset from numpy array
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, normalize=False):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.normalize = normalize


    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image/255.0
            image = (image-0.5)/0.5

        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

    def __len__(self):
        return self.n_images


################################################################################
def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    #plt.title('Training Loss')
    plt.savefig(filename)


################################################################################
# Convenience function to count the number of parameters in a module
def count_parameters(module, verbose=True):
    num_parameters = sum([p.data.nelement() for p in module.parameters()])
    if verbose:
        print('Number of parameters: {}'.format(num_parameters))
    return num_parameters


################################################################################
# predict class labels
# compute entropy of class labels; labels is a numpy array
def compute_entropy(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def predict_class_labels(net, images, batch_size=500, verbose=False, num_workers=0):
    net = net.cuda()
    net.eval()

    n = len(images)
    if batch_size>n:
        batch_size=n
    dataset_pred = IMGs_dataset(images, normalize=False)
    dataloader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_labels_pred = np.zeros(n+batch_size)
    with torch.no_grad():
        nimgs_got = 0
        if verbose:
            pb = SimpleProgressBar()
        for batch_idx, batch_images in enumerate(dataloader_pred):
            batch_images = batch_images.type(torch.float).cuda()
            batch_size_curr = len(batch_images)

            outputs,_ = net(batch_images)
            _, batch_class_labels_pred = torch.max(outputs.data, 1)
            class_labels_pred[nimgs_got:(nimgs_got+batch_size_curr)] = batch_class_labels_pred.detach().cpu().numpy().reshape(-1)

            nimgs_got += batch_size_curr
            if verbose:
                pb.update((float(nimgs_got)/n)*100)
        #end for batch_idx
    class_labels_pred = class_labels_pred[0:n]
    return class_labels_pred



# ################################################################################
# # create the end points of vicinities
# def create_endpoints(labels, min_nsamp):
#     ## input:
#     ## labels: numpy array, normalized labels ([0,1]) in the training set
#     ## min_nsamp: the minimum number of samples in the vicinity

#     ## output:
#     ## A dict: keys are distinct real labels and values are the endpoints

#     assert 0<=labels.min()<=1 and 0<=labels.max()<=1
#     assert min_nsamp>=1

#     unique_labels = np.sort(np.array(list(set(labels)))) ##sorted unique labels
#     num_unique_labels = len(unique_labels)
#     nsamp_per_label = [] ##number of images for each distinct label
#     for i in range(num_unique_labels):
#         num_i = len(np.where(labels==unique_labels[i])[0])
#         nsamp_per_label.append(num_i)
#     nsamp_per_label = np.array(nsamp_per_label)

#     ## the difference between two continuous distinct labels
#     diff_list = []
#     for i in range(1,num_unique_labels):
#         diff_list.append(unique_labels[i] - unique_labels[i-1])


#     endpoints_pairs = [] #[(start, end),...]
#     for i in range(num_unique_labels):
#         nsamp_i = nsamp_per_label[i] #the number of real images with labels equal to unique_labels[i]
#         start_i = end_i = unique_labels[i]
         
#         if i==0:
#             right_i = i
#             radius_i = diff_list[right_i]
#             while nsamp_i<min_nsamp:
#                 end_i = min(unique_labels[-1], unique_labels[i] + radius_i)
#                 nsamp_i = len(np.where((labels>=start_i)*(labels<=end_i))[0])
#                 if right_i+1>len(diff_list)-1: #termination condition
#                     break
#                 else:
#                     right_i = right_i+1
#                 radius_i = radius_i + diff_list[right_i] ##update radius
#             ##end while nsamp
#         elif 0<i<num_unique_labels-1:
#             left_i, right_i = i-1, i
#             radius_i = max(diff_list[left_i], diff_list[right_i])
#             while nsamp_i<min_nsamp:
#                 start_i = max(unique_labels[0], unique_labels[i] - radius_i)
#                 end_i = min(unique_labels[-1], unique_labels[i] + radius_i)
#                 nsamp_i = len(np.where((labels>=start_i)*(labels<=end_i))[0])
#                 if left_i-1 >= 0:
#                     left_i = left_i-1 #move the left pointer to the left
#                     diff_left = diff_list[left_i]
#                 else:
#                     diff_left = 0
#                 if right_i+1 <= len(diff_list)-1:
#                     right_i = right_i+1 #move the right pointer to the right
#                     diff_right = diff_list[right_i]
#                 else:
#                     diff_right = 0
#                 if left_i==0 and right_i==len(diff_list)-1 and diff_left==0 and diff_right==0: #termination condition
#                     break
#                 radius_i = radius_i + max(diff_left, diff_right) ##update radius
#             ##end while nsamp
#         else:
#             left_i = len(diff_list)-1
#             radius_i = diff_list[left_i]
#             while nsamp_i<min_nsamp:
#                 start_i = max(unique_labels[0], unique_labels[i] - radius_i)
#                 nsamp_i = len(np.where((labels>=start_i)*(labels<=end_i))[0])
#                 if left_i-1<0: #termination condition
#                     break
#                 else:
#                     left_i = left_i-1
#                 radius_i = radius_i + diff_list[left_i] ##update radius
#             ##end while nsamp
#         endpoints_pairs.append([start_i, end_i])
#     ##end for i

#     label_to_endpoints = dict()
#     for i in range(num_unique_labels):
#         label_to_endpoints[unique_labels[i]] = endpoints_pairs[i]

#     return label_to_endpoints, unique_labels



# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]
