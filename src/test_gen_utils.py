from gen_utils import *
import torch
import torch.nn as nn
from torch.autograd import Variable as V

import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image

L0 = nn.MSELoss()
L1 = nn.MSELoss(size_average=False)
L2 = LqLoss(q=2, eps=0)
L3 = LqLoss(q=2, eps=0, size_average=False)
L4 = LqLoss(q=1.8, eps=0.01)
L5 = LqLoss(q=1.8, eps=0.01, size_average=False)


x1, x2 = V(torch.rand(1,1,2,2)), V(torch.rand(1,1,2,2))

y0 = L0(x1, x2)
y1 = L1(x1, x2)

y2 = L2(x1,x2)
y3 = L3(x1,x2)
y4 = L4(x1,x2)
y5 = L5(x1,x2)

print('y0 ', y0)
print('y1 ', y1)
print('y2 ', y2)
print('y3 ', y3)
print('y4 ', y4)
print('y5 ', y5)


img = Image.open('/home/uddeshya/Desktop/projectX/RS3RGAN/data/mini_sample_dataset/train/unlabeled/Normal_001_32902_91875_0_cropped_slide.png')
img = np.array(img)
plt.figure()
plt.imshow(img)
plt.show()

n_img = add_gaussian_noise(img, 20)
plt.figure()
plt.imshow(n_img)
plt.show()

while(True):
	pass