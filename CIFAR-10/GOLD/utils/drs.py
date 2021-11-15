import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from tqdm import tqdm, trange

from utils import make_z, make_y


def get_score_stats(netG, netD, sample=50000):
	score_D = sample_scores(netG, netD, wd=1, wc=0, sample_size=sample)
	score_C = sample_scores(netG, netD, wd=0, wc=1, sample_size=sample)

	M = np.exp(np.max(score_D))
	w = np.std(score_D) / np.sqrt(np.mean(np.square(score_C)))

	return M, w


def sample_scores(netG, netD, nz=128, ny=10, wd=1, wc=1, sample_size=50000, batch_size=100):
	netG = netG.cuda()
	netD = netD.cuda()
	netG.eval()
	netD.eval()
	scores = []
	for i in range(sample_size // batch_size):
		z = make_z(batch_size, nz).cuda()
		y = make_y(batch_size, ny).cuda()
		with torch.no_grad():
			x = netG(z, y)
			s = gold(netD, x, y, wd, wc)
		scores.append(s)
	scores = np.concatenate(scores, axis=0)
	return scores


def gold(netD, x, y, wd=1, wc=1, verbose=False):
	netD = netD.cuda()
	netD.eval()
	with torch.no_grad():
		out_D, out_C = netD(x)  # B x 1, B x nc

	score_D = out_D.view(-1) * wd
	out_C = torch.softmax(out_C, dim=1)
	out_C = out_C[torch.arange(len(out_C)), y]
	score_C = torch.log(out_C) * wc

	if verbose:
		plt.hist(score_D.cpu().numpy())
		plt.hist(score_C.cpu().numpy())

	return (score_D + score_C).cpu().numpy()


def drs(netG, netD, num_samples=1000, perc=10, nz=128, ny=10, batch_size=100, eps=1e-6):
	netG = netG.cuda()
	netD = netD.cuda()
	netG.eval()
	netD.eval()
	
	M, w = get_score_stats(netG, netD)
	ones = np.ones(batch_size).astype('int')

	images = [[] for _ in range(ny)]

	for cls in trange(ny):
		while len(images[cls]) < num_samples:
			z = make_z(batch_size, nz).cuda()
			y = make_y(batch_size, ny, cls).cuda()
			with torch.no_grad():
				x = netG(z, y)
				r = np.exp(gold(netD, x, y, 1, w))

			p = np.minimum(ones, r/M)
			f = np.log(p + eps) - np.log(1 - p + eps)  # inverse sigmoid
			f = (f - np.percentile(f, perc))
			p = [1 / (1 + math.exp(-x)) for x in f]  # sigmoid
			accept = np.random.binomial(ones, p)

			for i in range(batch_size):
				if accept[i] and len(images[cls]) < num_samples:
					images[cls].append(x[i].detach().cpu())

	images = torch.stack([x for l in images for x in l])
	labels = []
	for i in range(ny):
		for _ in range(num_samples):
			labels.append(i)
	labels = torch.tensor(labels).type(torch.long)

	images = images.numpy()
	labels = labels.numpy()
	
	assert images.max()<=1.0 and images.min()>=-1.0
	images = ((images*0.5+0.5)*255).astype(np.uint8)

	return images, labels
