import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from PIL import Image
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

### import my stuffs ###
from models import *

"""""""""""""""	
Generate noise
"""""""""""""""

def make_z(size, nz):
	"""Return B x nz noise vector"""
	return torch.randn(size, nz)  # B x nz

def make_y(size, ny, value=None):
	"""Return B condition vector"""
	if value is None:
		return torch.randint(ny, [size]).long()  # B (random value)
	else:
		return torch.LongTensor(size).fill_(value)  # B (given value)

def make_fixed_z(size, nz, ny):
	"""Return (B * ny) x nz noise vector (for visualization)"""
	z = make_z(size, nz)  # B x nz
	return torch.cat([z] * ny, dim=0)  # (B x ny) x nz

def make_fixed_y(size, ny):
	"""Return (B * ny) condition vector (for visualization)"""
	y = [torch.LongTensor(size).fill_(i) for i in range(ny)]  # list of B tensors
	return torch.cat(y, dim=0)  # (B * ny)


"""""""""""""""	
Helper functions (I/O)
"""""""""""""""

def count_classes(dataset, class_num):
	count = [0] * class_num
	for _, y in dataset:
		count[y] += 1
	return count

def save_to_logger(logger, info, step):
	for key, val in info.items():
		if isinstance(val, np.ndarray):
			logger.image_summary(key, val, step)
		else:
			logger.scalar_summary(key, val, step)

def normalize_info(info):
	num = info.pop('num')
	for key, val in info.items():
		info[key] /= num
	return info

def gold_score(netD, x, y, eps=1e-6):
	out_D, out_C = netD(x)  # B x 1, B x nc
	out_C = torch.softmax(out_C, dim=1)  # B x nc
	score_C = torch.log(out_C[torch.arange(len(out_C)), y] + eps)  # B
	return out_D.view(-1) + score_C  # B

def entropy(outs, eps=0):
	probs = F.softmax(outs, dim=1)  # B x nc
	entropy = -(probs * torch.log(probs + eps)).sum(-1)  # B
	return entropy  # B

def accuracy(out, tgt):
	_, pred = out.max(1)
	acc = pred.eq(tgt).sum().item() / len(out)
	return acc

def to_numpy_image(x):
	# convert torch tensor [-1,1] to numpy image [0,255]
	x = x.cpu().numpy().transpose(0, 2, 3, 1)  # C x H x W -> H x W x C
	x = ((x + 1) / 2).clip(0, 1)  # [-1,1] -> [0,1]
	x = (x * 255).astype(np.uint8)  # uint8 numpy image
	return x


### Progress Bar
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
    def __init__(self, images, labels=None, transform=None):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.transform = transform

    def __getitem__(self, index):

        ## for RGB only
        image = self.images[index]
        if self.transform is not None:
            image = np.transpose(image, (1, 2, 0)) #C * H * W ---->  H * W * C
            image = Image.fromarray(np.uint8(image), mode = 'RGB') #H * W * C
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[index]

            return image, label

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