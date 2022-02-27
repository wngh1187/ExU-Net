import torch
import torch.nn as nn
from models.ResNetBlocks import *


class UNet(nn.Module): 
	""" 
	UNet-based system
	"""
	def __init__(self, args):
		super(UNet, self).__init__()
		self.l_channel = args.model['l_channel']
		self.l_num_convblocks = args.model['l_num_convblocks']
		self.code_dim = args.model['code_dim']
		self.stride = args.model['l_stride']
		self.first_kernel_size = args.model['first_kernel_size']
		self.first_stride_size = args.model['first_stride_size']
		self.first_padding_size = args.model['first_padding_size']

		self.inplanes   = self.l_channel[0]
		self.instancenorm   = nn.InstanceNorm1d(args.nfilts)
		
		self.conv1 = nn.Conv2d(1, self.l_channel[0] , kernel_size=self.first_kernel_size, stride=self.first_stride_size, padding=self.first_padding_size,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(self.l_channel[0])
		self.relu = nn.ReLU(inplace=True)

		self.n_level = len(self.l_channel)
		
		###############################
		# Description of Encoder Path #
		###############################
		for i in range(0, self.n_level):
			setattr(self, 'd_res_{}'.format(i+1), self._make_d_layer(SEBasicBlock, self.l_channel[i], self.l_num_convblocks[i], self.stride[i]))

		######################################################
		# speaker embedding layer and speaker identification #
		######################################################
		final_dim = self.l_channel[-1]
		
		self.attention = nn.Sequential(
			nn.Conv1d(final_dim, final_dim//8, kernel_size=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(final_dim//8),
			nn.Conv1d(final_dim//8, final_dim, kernel_size=1), 
			nn.Softmax(dim=-1),
		)
		
		self.bn_agg = nn.BatchNorm1d(final_dim * 2)

		self.fc = nn.Linear(final_dim*2, self.code_dim)
		self.bn_code = nn.BatchNorm1d(self.code_dim)

		###############################
		# Description of Decoder Path #
		###############################
		for i in range(0, self.n_level):
			setattr(self, 'u_res_{}'.format(i), self._make_u_layer(SEBasicBlock, self.l_channel[-i-1], self.l_num_convblocks[-i-1]))
			if i == 0: setattr(self, 'uconv_{}'.format(i), nn.Conv2d(self.l_channel[-i-1]*2, self.l_channel[-i-2], kernel_size=1, bias=False))
			elif i == self.n_level-1 : setattr(self, 'uconv_{}'.format(i), nn.Conv2d(self.l_channel[0]*2, self.l_channel[0], kernel_size=1, bias=False))
			else: setattr(self, 'uconv_{}'.format(i), nn.ConvTranspose2d(self.l_channel[-i-1]*2, self.l_channel[-i-2], kernel_size=2, stride=2, bias=False))
		setattr(self, 'uconv_{}'.format(self.n_level), nn.ConvTranspose2d(self.l_channel[0]*2, 1, kernel_size=(2,1), stride = (2,1), bias=False))
			

		self.lRelu   = nn.LeakyReLU(negative_slope=0.2)
		self.relu    = nn.ReLU()
		self.drop    = nn.Dropout(0.5)
		
	def _make_d_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _make_u_layer(self, block, planes, blocks, stride=1):
		downsample = None

		layers = []
		layers.append(block(planes, planes, stride, downsample))
		for i in range(1, blocks):
			layers.append(block(planes, planes))

		return nn.Sequential(*layers)

	def forward(self, x, only_code=False):
		
		x = self.instancenorm(x).unsqueeze(1).detach()
		d_dx = {}
		u_dx = {}

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		d_dx['0'] = x

		# +++++++++++++++++++ Encoder Path  +++++++++++++++++++++ #
		for i in range(1, self.n_level+1):
			d_dx['%d'%i] = getattr(self, 'd_res_{}'.format(i))(d_dx['%d'%(i-1)])
		
		x = d_dx['%d'%(self.n_level)]

		bs, ch, _, _ = x.size()
		x = x.reshape(bs, ch, -1)

		w = self.attention(x)
		m = torch.sum(x * w, dim=-1)
		s = torch.sqrt((torch.sum((x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
		x = torch.cat([m, s], dim=1)
		x =	self.bn_agg(x)

		code = self.fc(x)
		code = self.bn_code(code)
	
		if only_code: return code

		# +++++++++++++++++++ Decoder Path  +++++++++++++++++++++ #
		u_input = d_dx['%d'%(self.n_level)]
		for i in range(0, self.n_level):	
			u_dx['%d'%i] = getattr(self, 'u_res_{}'.format(i))((u_input))
			u_input = torch.cat((u_dx['%d'%(i)], d_dx['%d'%(self.n_level-i)]), 1)
			u_input = getattr(self, 'uconv_{}'.format(i))((u_input))
			
		output = torch.cat((u_input, d_dx['0']), 1)
		output = getattr(self, 'uconv_{}'.format(self.n_level))((output))
		
		return code, output

def get_unet(args):
	model = UNet(args)
	return model