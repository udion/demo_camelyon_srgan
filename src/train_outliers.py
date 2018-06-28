import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboard_logger import configure, log_value

from models import Generator, Discriminator, FeatureExtractor
# from utils import Visualizer
import PIL
from PIL import ImageFilter
from PIL import Image
import random
from gen_utils import add_gaussian_noise, probab_salt_n_pepper, random_blackout
import numpy as np

import datetime
timestamp = str(datetime.datetime.now()).replace(' ', '_')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='folder', help='folder')
parser.add_argument('--dataroot', type=str, default='../data/outliers/train', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=6, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='', help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='../checkpoints', help='folder to output model checkpoints')

opt = parser.parse_args()
opt.out = opt.out+'/'+timestamp
print(opt)

try:
	os.makedirs(opt.out)
except OSError:
	pass

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize*(2**(opt.upSampling//2))),
								transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
								std = [0.229, 0.224, 0.225])

#this will help generate the smaller resolution version of the images, first convert it 
#to 0 1 scale then add gaussian noise then scale and (then redo the standard normalisation)
scale = transforms.Compose([transforms.ToPILImage(),
							lambda l: random_blackout(l, opt.imageSize*4, opt.imageSize*4),
							lambda l : probab_salt_n_pepper(l, 6, 0.65),
							lambda l: l.filter(ImageFilter.GaussianBlur(radius=2)),
							transforms.Resize(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
		])

if opt.dataset == 'folder':
	# folder dataset
	dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
else:
	print('other than folder option not supported')
	sys.exit(0)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

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

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
if opt.cuda:
	G.cuda()
	D.cuda()
	FE.cuda()
	content_criterion.cuda()
	adversarial_criterion.cuda()
	ones_const = ones_const.cuda()

optim_generator = optim.Adam(G.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(D.parameters(), lr=opt.discriminatorLR)

configure('../logs/'+timestamp, flush_secs=5)
# visualizer = Visualizer(image_size=opt.imageSize*opt.upSampling)

low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# Pre-train generator using raw MSE loss
if opt.generatorWeights == '':
	print('Generator pre-training')
	for epoch in range(2):
		mean_generator_content_loss = 0.0

		for i, data in enumerate(dataloader):
			# Generate data
			high_res_real, _ = data

			# Downsample images to low resolution
			n_j = high_res_real.size()[0]
			low_res = torch.FloatTensor(n_j, 3, opt.imageSize, opt.imageSize)
			for j in range(n_j):
				low_res[j] = scale(high_res_real[j])
				# print('dbg ', low_res.shape)
				high_res_real[j] = normalize(high_res_real[j])

			# Generate real and fake inputs
			if opt.cuda:
				high_res_real = Variable(high_res_real.cuda())
				high_res_fake = G(Variable(low_res).cuda())
			else:
				high_res_real = Variable(high_res_real)
				high_res_fake = G(Variable(low_res))
			# print('dbg ', high_res_fake.shape)
			
			######### Train generator #########
			G.zero_grad()
			# print('real shape, fake shape : ', high_res_real.size(), high_res_fake.size())
			generator_content_loss = content_criterion(high_res_fake, high_res_real)
			mean_generator_content_loss += generator_content_loss.data[0]

			generator_content_loss.backward()
			optim_generator.step()

			######### Status and display #########
			sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, 2, i, len(dataloader), generator_content_loss.data[0]))
			# visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)

		sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch, 2, i, len(dataloader), mean_generator_content_loss/len(dataloader)))
		log_value('generator_mse_loss', mean_generator_content_loss/len(dataloader), epoch)

	# Do checkpointing
	torch.save(G.state_dict(), '%s/generator_pretrain.pth' % opt.out)

# SRGAN training
optim_generator = optim.Adam(G.parameters(), lr=opt.generatorLR) #*0.1
optim_discriminator = optim.Adam(D.parameters(), lr=opt.discriminatorLR)

print('SRGAN training ...')
for epoch in range(opt.nEpochs):
	mean_generator_content_loss = 0.0
	mean_generator_adversarial_loss = 0.0
	mean_generator_total_loss = 0.0
	mean_discriminator_loss = 0.0

	for i, data in enumerate(dataloader):
		# Generate data
		high_res_real, _ = data

		n_j = high_res_real.size()[0]
		low_res = torch.FloatTensor(n_j, 3, opt.imageSize, opt.imageSize)
		ones_const = Variable(torch.ones(n_j, 1))
		if opt.cuda:
			ones_const = ones_const.cuda()
		# Downsample images to low resolution
		# Also corrupt some high resolution images in each batch
		n_corrupt_per_batch = 1
		corrupt_indxs = random.sample(range(n_j), n_corrupt_per_batch)
		for j in range(n_j):
			low_res[j] = scale(high_res_real[j])
			if j in corrupt_indxs:
				nimg = add_gaussian_noise(high_res_real[j], sig=20)
				high_res_real[j] = torch.from_numpy(nimg)
			high_res_real[j] = normalize(high_res_real[j])

		# Generate real and fake inputs
		if opt.cuda:
			high_res_real = Variable(high_res_real.cuda())
			high_res_fake = G(Variable(low_res).cuda())
			target_real = Variable(torch.rand(n_j,1)*0.5 + 0.7).cuda()
			target_fake = Variable(torch.rand(n_j,1)*0.3).cuda()
		else:
			high_res_real = Variable(high_res_real)
			high_res_fake = G(Variable(low_res))
			target_real = Variable(torch.rand(n_j,1)*0.5 + 0.7)
			target_fake = Variable(torch.rand(n_j,1)*0.3)
		
		######### Train discriminator #########
		D.zero_grad()

		discriminator_loss = adversarial_criterion(D(high_res_real), target_real) + adversarial_criterion(D(Variable(high_res_fake.data)), target_fake)
		
		mean_discriminator_loss += discriminator_loss.data[0]
		
		discriminator_loss.backward()
		optim_discriminator.step()

		######### Train generator #########
		G.zero_grad()

		real_features = Variable(FE(high_res_real).data)
		fake_features = FE(high_res_fake)

		generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
		mean_generator_content_loss += generator_content_loss.data[0]
		
		generator_adversarial_loss = adversarial_criterion(D(high_res_fake), ones_const)
		mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

		generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
		mean_generator_total_loss += generator_total_loss.data[0]
		
		generator_total_loss.backward()
		optim_generator.step()   
		
		######### Status and display #########
		sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch, opt.nEpochs, i, len(dataloader),
		discriminator_loss.data[0], generator_content_loss.data[0], generator_adversarial_loss.data[0], generator_total_loss.data[0]))
		# visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)

	sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, opt.nEpochs, i, len(dataloader),
	mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
	mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))

	log_value('generator_content_loss', mean_generator_content_loss/len(dataloader), epoch)
	log_value('generator_adversarial_loss', mean_generator_adversarial_loss/len(dataloader), epoch)
	log_value('generator_total_loss', mean_generator_total_loss/len(dataloader), epoch)
	log_value('discriminator_loss', mean_discriminator_loss/len(dataloader), epoch)

	# Do checkpointing
	torch.save(G.state_dict(), '%s/generator_final.pth' % opt.out)
	torch.save(D.state_dict(), '%s/discriminator_final.pth' % opt.out)

# # Avoid closing
# print('done training... hit CTRL+C')
# while True:
# 	pass
