import math
import random
import numpy as np

import torch
import torch.nn as nn

__all__=['duplicate', 'subtensor', 'linspace_crop', 'rand_crop']

def duplicate(x, size, dim=-1):
    """duplicate tensor in given dimension 
    until x is larger than size

    params
        x       - tensor to duplicate
        size    - minimum size
        dim     - target dimension
    """
    length = x.size(dim)
    x = torch.cat(tuple(
        x for i in range(math.ceil(size / length))), dim=dim)
    return x 
        
def subtensor(x, start, size, dim=-1):
    """    
    Param
        x       - tensor to crop
        start   - start index
        size    - size of tensor
        dim     - target dimension
    """
    dim_indexes = [i for i in range(len(x.size()))]
    temp = dim_indexes[0]
    dim_indexes[0] = dim_indexes[dim]
    dim_indexes[dim] = temp

    x = x.permute(*dim_indexes) # [..., default_size] -> [defalut_size, ...]
    x = x[start:start + size]
    x = x.permute(*dim_indexes) # [defalut_size, ...] -> [..., default_size]
    
    return x

def linspace_crop(x, num, size, dim=-1):
    """linspace crop for Test Time Augmentation(TTA)

    Param
        x       - tensor to crop
        num     - linspace num
        size    - size of tensor
        dim     - target dimension
    """
    # duplicate
    if x.size(dim) < size:
        x = duplicate(x, size, dim)

    # calculate index
    index = np.linspace(0, x.size(dim) - size, num=num)

    # init buffer
    buffer_size = list(x.size())
    buffer_size[dim] = size
    buffer_size = tuple([len(index)] + buffer_size)
    tensor_buffer = torch.zeros(buffer_size, device='cuda' if x.is_cuda else 'cpu')

    # slice
    for i in range(num):
        tensor_buffer[i] = subtensor(x, int(index[i]), size, dim)

    return tensor_buffer

def rand_crop(x, size, dim=-1):
    """random crop for data augmentation

    Param
        x       - tensor to crop
        size    - size of tensor
        dim     - target dimension
    """
    if x.size(dim) < size:
        x = duplicate(x, size, dim)
    index = random.randint(0, x.size(dim) - size)
    return subtensor(x, index, size, dim)

def init_weights(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.0001)
	elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm1d):
		pass
	else:
		if hasattr(m, 'weight'):
			nn.init.kaiming_normal_(m.weight, a=0.01)
		else:
			pass