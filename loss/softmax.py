import torch.nn as nn

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
		super(LossFunction, self).__init__()

		self.test_normalize = True
		
		self.criterion  = nn.CrossEntropyLoss()
		self.fc 		= nn.Linear(nOut, nClasses, bias=True)

		print('Initialised Softmax Loss')

	def forward(self, x, label=None):

		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		return nloss