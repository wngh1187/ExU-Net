#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/clovaai/voxceleb_trainer/tree/master/loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):

	def __init__(self, gpu, init_w=10.0, init_b=-5.0, **kwargs):
		super(LossFunction, self).__init__()

		self.gpu = gpu
		self.w = nn.Parameter(torch.tensor(init_w))
		self.b = nn.Parameter(torch.tensor(init_b))
		self.w.requires_grad = True
		self.b.requires_grad = True 
		self.cce = nn.CrossEntropyLoss()

		print('Initialised AngleProto')

	def forward(self, out_anchor, out_positive):
		
		#assert x.size()[1] >= 2

		stepsize        = out_anchor.size()[0]

		cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
		torch.clamp(self.w, 1e-6)
		cos_sim_matrix = cos_sim_matrix * self.w + self.b
		
		label       = torch.from_numpy(np.asarray(range(0,stepsize))).cuda(self.gpu)
		criterion = self.cce
		loss       = criterion(cos_sim_matrix, label)

		return loss
