import argparse
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import Generator, Discriminator, FeatureExtractor

import PIL
from PIL import ImageFilter
from PIL import Image
import random
from gen_utils import add_gaussian_noise


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='folder', help='folder')
parser.add_argument('--dataroot', type=str, default='../data/small_splits/test', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=6, help='low to high resolution scaling factor')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='../checkpoints/generator_final.pth', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='../checkpoints/discriminator_final.pth', help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='model_outputs', help='name of the folder inside outputs to place the results')

opt = parser.parse_args()
print(opt)

try:
	os.makedirs('../outputs/{}/high_res_fake'.format(opt.out))
	os.makedirs('../outputs/{}/high_res_real'.format(opt.out))
	os.makedirs('../outputs/{}/low_res'.format(opt.out))
except OSError:
	pass


if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize*(2**(opt.upSampling//2))), transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

#this will help generate the smaller resolution version of the images, first convert it 
#to 0 1 scale then add gaussian noise then scale and (then redo the standard normalisation)
scale = transforms.Compose([transforms.ToPILImage(),
							lambda l: l.filter(ImageFilter.GaussianBlur(radius=2)),
							transforms.Resize(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
		])

# Equivalent to un-normalizing ImageNet (for correct visualization)
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

if opt.dataset == 'folder':
	# folder dataset
	dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
# elif opt.dataset == 'cifar10':
# 	dataset = datasets.CIFAR10(root=opt.dataroot, download=True, train=False, transform=transform)
# elif opt.dataset == 'cifar100':
# 	dataset = datasets.CIFAR100(root=opt.dataroot, download=True, train=False, transform=transform)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

G = Generator(10, opt.upSampling)
if opt.generatorWeights != '':
	G.load_state_dict(torch.load(opt.generatorWeights))
print(G)

D = Discriminator()
if opt.discriminatorWeights != '':
	D.load_state_dict(torch.load(opt.discriminatorWeights))
print(D)

# For the content loss
FE = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print(FE)
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

target_real = Variable(torch.ones(opt.batchSize,1))
target_fake = Variable(torch.zeros(opt.batchSize,1))

# if gpu is to be used
if opt.cuda:
	G.cuda()
	D.cuda()
	FE.cuda()
	content_criterion.cuda()
	adversarial_criterion.cuda()
	target_real = target_real.cuda()
	target_fake = target_fake.cuda()

low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

print('Test started...')
mean_generator_content_loss = 0.0
mean_generator_adversarial_loss = 0.0
mean_generator_total_loss = 0.0
mean_discriminator_loss = 0.0

# Set evaluation mode (not training)
G.eval()
D.eval()

for i, data in enumerate(dataloader):
	# Generate data
	high_res_real, _ = data

	# Downsample images to low resolution
	n_j = high_res_real.size()[0]
	low_res = torch.FloatTensor(n_j, 3, opt.imageSize, opt.imageSize)

	target_real = Variable(torch.ones(n_j,1))
	target_fake = Variable(torch.zeros(n_j,1))

	for j in range(n_j):
		low_res[j] = scale(high_res_real[j])
		high_res_real[j] = normalize(high_res_real[j])

	# Generate real and fake inputs
	if opt.cuda:
		high_res_real = Variable(high_res_real.cuda())
		high_res_fake = G(Variable(low_res).cuda())
		target_real = Variable(torch.ones(n_j,1).cuda())
		target_fake = Variable(torch.zeros(n_j,1).cuda())
	else:
		high_res_real = Variable(high_res_real)
		high_res_fake = G(Variable(low_res))
		target_real = Variable(torch.ones(n_j,1))
		target_fake = Variable(torch.zeros(n_j,1))
	
	######### Test discriminator #########

	discriminator_loss = adversarial_criterion(D(high_res_real), target_real) + adversarial_criterion(D(Variable(high_res_fake.data)), target_fake)
	mean_discriminator_loss += discriminator_loss.data[0]

	######### Test generator #########

	real_features = Variable(FE(high_res_real).data)
	fake_features = FE(high_res_fake)

	generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
	mean_generator_content_loss += generator_content_loss.data[0]
	generator_adversarial_loss = adversarial_criterion(D(high_res_fake), target_real)
	mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

	generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
	mean_generator_total_loss += generator_total_loss.data[0]

	######### Status and display #########
	sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (i, len(dataloader),
	discriminator_loss.data[0], generator_content_loss.data[0], generator_adversarial_loss.data[0], generator_total_loss.data[0]))

	for j in range(n_j):
		save_image(unnormalize(high_res_real.data[j]), '../outputs/{}/high_res_real/'.format(opt.out) + str(i*opt.batchSize + j) + '.png')
		save_image(unnormalize(high_res_fake.data[j]), '../outputs/{}/high_res_fake/'.format(opt.out) + str(i*opt.batchSize + j) + '.png')
		save_image(unnormalize(low_res[j]), '../outputs/{}/low_res/'.format(opt.out) + str(i*opt.batchSize + j) + '.png')

sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (i, len(dataloader),
mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))
