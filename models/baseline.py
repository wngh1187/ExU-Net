import torch
import torch.nn as nn
from models.ResNetBlocks import *

class Baseline(nn.Module): 
	""" 
	SEResNet
	"""
	def __init__(self, args):
		super(Baseline, self).__init__()
		self.l_channel = args.model['l_channel']
		self.l_num_convblocks = args.model['l_num_convblocks']
		self.code_dim = args.model['code_dim']
		self.l_stride = args.model['l_stride']
		self.first_kernel_size = args.model['first_kernel_size']
		self.first_stride_size = args.model['first_stride_size']
		self.first_padding_size = args.model['first_padding_size']
		
		self.inplanes   = self.l_channel[0]
		self.instancenorm   = nn.InstanceNorm1d(args.nfilts)
		
		self.conv1 = nn.Conv2d(1, self.l_channel[0] , kernel_size=self.first_kernel_size, stride=self.first_stride_size, padding=self.first_padding_size,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(self.l_channel[0])
		self.relu = nn.ReLU(inplace=True)

		self.layer1 = self._make_layer(SEBasicBlock, self.l_channel[0], self.l_num_convblocks[0], stride=self.l_stride[0])
		self.layer2 = self._make_layer(SEBasicBlock, self.l_channel[1], self.l_num_convblocks[1], stride=self.l_stride[1])
		self.layer3 = self._make_layer(SEBasicBlock, self.l_channel[2], self.l_num_convblocks[2], stride=self.l_stride[2])
		self.layer4 = self._make_layer(SEBasicBlock, self.l_channel[3], self.l_num_convblocks[3], stride=self.l_stride[3])

		final_dim = self.l_channel[-1]
		
		self.attention = nn.Sequential(
			nn.Conv1d(final_dim, final_dim//8, kernel_size=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(final_dim//8),
			nn.Conv1d(final_dim//8, final_dim, kernel_size=1), 
			nn.Softmax(dim=-1),
		)
		
		######################################################
		# speaker embedding layer and speaker identification #
		######################################################
		self.bn_agg = nn.BatchNorm1d(final_dim * 2)

		self.fc = nn.Linear(final_dim*2, self.code_dim)
		self.bn_code = nn.BatchNorm1d(self.code_dim)


		self.lRelu   = nn.LeakyReLU(negative_slope=0.2)
		self.relu    = nn.ReLU()
		self.drop    = nn.Dropout(0.5)
		
	def _make_layer(self, block, planes, blocks, stride=1):
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

	def forward(self, x, only_code=False):
		
		x = self.instancenorm(x).unsqueeze(1).detach()

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		bs, ch, _, _ = x.size()
		x = x.reshape(bs, ch, -1)

		w = self.attention(x)
		m = torch.sum(x * w, dim=-1)
		s = torch.sqrt((torch.sum((x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
		x = torch.cat([m, s], dim=1)
		x =	self.bn_agg(x)

		code = self.fc(x)
		code = self.bn_code(code)
		
		dummy = None
		if only_code: return code
		return code, dummy	
		
def get_baseline(args):
	model = Baseline(args)
	return model