import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce

import numpy as np
import random
import PIL
from PIL import ImageFilter
from PIL import Image

def Lq_norm(inputs, targets, q, eps_mod, size_average=True):
	#print('inputs, targets', inputs, targets)
	eps = eps_mod*Variable(torch.randn(inputs.size()))
	t = (inputs-targets)**2 + eps
	#print(t)
	t = t**(q/2)
	#print(t)
	t = torch.sum(t)
	#print(t)
	if size_average:
		import operator
		N = reduce(operator.mul, list(inputs.size()))
		t = t/N
	return t

class LqLoss(nn.Module):
	def __init__(self, q, eps, size_average=True, cuda_enabled=False):
		super(LqLoss, self).__init__()
		self.q = q
		self.eps = eps
		self.size_average = size_average
		self.cuda_enabled = cuda_enabled

	def forward(self, inputs, targets):
		eps = self.eps
		q = self.q
		size_average = self.size_average
		#print('inputs, targets ', inputs, targets)
		t = (inputs-targets)**2 + eps
		#print(t)
		t = t**(q/2)
		#print(t)
		t = torch.sum(t)
		#print(t)
		if size_average:
			import operator
			N = reduce(operator.mul, list(inputs.size()))
			t = t/N
		return t

def add_gaussian_noise(img, sig=1, mu=0):
	'''
		img is PILimage
	'''
	img = np.asarray(img, dtype='float')
	d1, d2, d3 = img.shape
	noise_image = np.random.randn(d1, d2, d3)*sig
	img += noise_image
	mmin = np.min(img)
	img = img-mmin
	mmax = np.max(img)
	img /= mmax
	return img

def salt_n_pepper(img, pad):
    nimg = np.array(img)
    H,W,C = nimg.shape
    noise = np.random.randint(pad, size = (H, W, C))
    nimg = np.where(noise == 0, 0, nimg)
    nimg = np.where(noise == pad-1, 255, nimg)
    nimg = np.asarray(nimg, dtype='uint8')
    img = PIL.Image.fromarray(nimg)
    return img

def probab_salt_n_pepper(img, pad, pthresh):
	v = random.random()
	if v >= pthresh:
		nimg = np.array(img)
		H,W,C = nimg.shape
		noise = np.random.randint(pad, size = (H, W, C))
		nimg = np.where(noise == 0, 0, nimg)
		nimg = np.where(noise == pad-1, 255, nimg)
		nimg = np.asarray(nimg, dtype='uint8')
		img = PIL.Image.fromarray(nimg)
	return img

def random_blackout(img, h, w):
    nimg = np.array(img)
    H,W,C = nimg.shape[0], nimg.shape[1], nimg.shape[2]
    r,c = random.randint(0, H-1), random.randint(0, W-1)
    r1 = min(H, r+h)
    c1 = min(W, c+w)
    b = nimg[r:r1][c:c1][:]
    nimg[r:r1, c:c1, :] = np.zeros((r1-r, c1-c,C), dtype='uint8')
    img = PIL.Image.fromarray(nimg)
    return img

