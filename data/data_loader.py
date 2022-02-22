import math
import torch
import torch.utils.data as td
import soundfile as sf
import numpy as np
import warnings

import utils.util as util
from data.musan import MusanNoise

def get_loaders(args, vox1):
	train_set = TrainSet(args, vox1)
	train_set_sampler = Voxceleb_sampler(dataset=train_set, nb_utt_per_spk=args['nb_utt_per_spk'], max_seg_per_spk=args['max_seg_per_spk'], batch_size=args['batch_size'])
	enrollment_set = EnrollmentSet(args, vox1)
	enrollment_set_sampler = td.DistributedSampler(enrollment_set, shuffle=False)

	train_loader = td.DataLoader(
		train_set,
		batch_size=args['batch_size'],
		pin_memory=True,
		num_workers=args['num_workers'],
		sampler=train_set_sampler
	)

	enrollment_loader = td.DataLoader(
		enrollment_set,
		batch_size=1,
		pin_memory=True,
		num_workers=args['num_workers'],
		sampler=enrollment_set_sampler
	)
	
	return train_set, train_loader, enrollment_set, enrollment_loader

class TrainSet(td.Dataset):
	def __init__(self, args, vox1):
		self.items = vox1.train_set
		
		# set label
		count = 0
		self.labels = {}
		for spk in vox1.train_speakers:
			self.labels[spk] = count
			count += 1
	
		# crop size
		self.crop_size = args['winlen'] + (args['winstep'] * (args['train_frame'] - 1))
	
		# musan
		self.musan = MusanNoise(f'{args["path_musan"]}_split/train')

		#####
		# for sampler
		self.utt_per_spk = {}
		self.revised_utts = []
		self.revised_labels = []
		for idx, line in enumerate(vox1.train_set):
			label = self.labels[line.path.split("/")[4]]
			if label not in self.utt_per_spk:
				self.utt_per_spk[label] = []
			self.utt_per_spk[label].append(idx)

	def __len__(self):
		return len(self.items)

	def __getitem__(self, indices):
		utts = []
		for i, index in enumerate(indices):
			item = self.items[index]

			# read wav
			data, _ = sf.read(item.path)
			
			# rand crop
			data = torch.from_numpy(data)
			data = util.rand_crop(data, self.crop_size)
			if i==0: utts.append(data)
			else:
				referance_utt = data.clone().detach()
				# noise injection
				noise_utt = self.musan(data)
				utts.append(noise_utt)
		return utts[0], utts[1], referance_utt, self.labels[item.speaker]


class Voxceleb_sampler(torch.utils.data.DistributedSampler):
	"""
	Acknowledgement: Github project 'clovaai/voxceleb_trainer'.
	link: https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py
	Adjusted for RawNeXt
	"""
	def __init__(self, dataset, nb_utt_per_spk, max_seg_per_spk, batch_size):
		# distributed settings
		if not torch.distributed.is_available():
			raise RuntimeError("Requires distributed package.")
		self.nb_replicas = torch.distributed.get_world_size()
		self.rank = torch.distributed.get_rank()
		self.epoch = 0

		# sampler config
		self.dataset = dataset
		self.utt_per_spk = dataset.utt_per_spk
		self.nb_utt_per_spk = nb_utt_per_spk
		self.max_seg_per_spk = max_seg_per_spk
		self.batch_size = batch_size
		self.nb_samples = int(
			math.ceil(len(dataset) / self.nb_replicas)
		)  
		self.total_size = (
			self.nb_samples * self.nb_replicas
		) 
		self.__iter__() 

	def __iter__(self):
		
		np.random.seed(self.epoch)

		# speaker ids
		spk_indices = np.random.permutation(list(self.utt_per_spk.keys()))

		# pair utterances by 2
		# list of list
		lol = lambda lst: [lst[i : i + self.nb_utt_per_spk] for i in range(0, len(lst), self.nb_utt_per_spk)]

		flattened_list = []
		flattened_label = []

		# Data for each class
		for findex, key in enumerate(spk_indices):
			# list, utt keys for one speaker
			utt_indices = self.utt_per_spk[key]
			# number of pairs of one speaker's utterances
			nb_seg = round_down(min(len(utt_indices), self.max_seg_per_spk), self.nb_utt_per_spk)
			# shuffle -> make to pairs
			rp = lol(np.random.permutation(len(utt_indices))[:nb_seg])
			flattened_label.extend([findex] * (len(rp)))
			for indices in rp:
				flattened_list.append([utt_indices[i] for i in indices])
		# print("a", np.array(flattened_list).shape) # a (562675, 2)
		# print("b", np.array(flattened_label).shape) # b (562675,)
		# data in random order
		mixid = np.random.permutation(len(flattened_label))
		mixlabel = []
		mixmap = []

		# prevent two pairs of the same speaker in the same batch
		for ii in mixid:
			startbatch = len(mixlabel) - (
				len(mixlabel) % (self.batch_size * self.nb_replicas)
			)
			if flattened_label[ii] not in mixlabel[startbatch:]:
				mixlabel.append(flattened_label[ii])
				mixmap.append(ii)
		it = [flattened_list[i] for i in mixmap]

		# adjust mini-batch-wise for DDP
		nb_batch, leftover = divmod(len(it), self.nb_replicas * self.batch_size)
		if leftover != 0:
			warnings.warn(
				"leftover:{} in sampler, epoch:{}, gpu:{}, cropping..".format(
					leftover, self.epoch, self.rank
				)
			)
			it = it[: self.nb_replicas * self.batch_size * nb_batch]
		_it = []
		for idx in range(
			self.rank * self.batch_size, len(it), self.nb_replicas * self.batch_size
		):
			_it.extend(it[idx : idx + self.batch_size])
		it = _it
		self._len = len(it)  # print("nb utt per GPU", self._len) # 138700 for 4GPU

		return iter(it)

	def __len__(self):
		return self._len

class EnrollmentSet(td.Dataset):
	@property
	def Key(self):
		return self.key
	@Key.setter
	def Key(self, value):
		self.key = value

	def __init__(self, args, vox1):
		self.key = 'clean'
		self.items = vox1.enrollment_set
		self.crop_size = args['winlen'] + (args['winstep'] * (args['test_frame'] - 1))
		
	def __len__(self):
		return len(self.items[self.Key])

	def __getitem__(self, index):
		item = self.items[self.Key][index]

		# read wav
		wav, _ = sf.read(item.path)

		if len(wav) < self.crop_size:
			nb_dup = int(self.crop_size / len(wav)) + 1
			wav = wav.repeat(nb_dup)[:self.crop_size]

		buffer = []
		index = np.arange(0, len(wav) - self.crop_size, self.crop_size)
		
		for i in range(len(index)):
			start = int(index[i])
			end = start + self.crop_size	
			buffer.append(wav[start:end])
		buffer.append(wav[-self.crop_size:])

		data = np.stack(buffer)
	
		return data, item.key
	
def round_down(num, divisor):
	return num - (num % divisor)