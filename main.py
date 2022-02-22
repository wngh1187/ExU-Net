import os
import random
import importlib
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import arguments
import trainers.train as train
import data.data_loader as data
from data.voxceleb1 import VoxCeleb1
from utils.util import init_weights
from utils.summary import summary_string
from models.deep_res_znet import SE_ResZNet
from log.controller import LogModuleController
from data.preprocessing import DataPreprocessor
from speech_features.log_melspectrogram import LogMelspectrogram

def set_experiment_environment(args):
	# reproducible
	random.seed(args['rand_seed'])
	np.random.seed(args['rand_seed'])
	torch.manual_seed(args['rand_seed'])
	torch.backends.cudnn.deterministic = args['flag_reproduciable']
	torch.backends.cudnn.benchmark = not args['flag_reproduciable']

	# DDP env
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '4021'
	args['rank'] = args['process_id']
	args['device'] = 'cuda:' + args['gpu_ids'][args['process_id']]
	torch.distributed.init_process_group(
			backend='nccl', world_size=args['world_size'], rank=args['rank'])

def run(process_id, args, experiemt_args):
	# check parent process
	args['process_id'] = process_id
	args['flag_parent'] = process_id == 0
	
	# experiment environment
	set_experiment_environment(args)
	trainer = train.ModelTrainer()
	trainer.args = args
	
	# logger
	if args['flag_parent']:
		logger = LogModuleController.Builder(args['name'], args['project']
		).tags(args['tags']
		).description(args['description']
		).save_source_files(args['path_scripts']
		).use_local(args['path_logging']
		).use_wandb(args['wandb_group'], args['wandb_entity']
		).build()
		trainer.logger = logger
	
	# dataset
	trainer.vox1 = VoxCeleb1(
		args['path_vox1_train'], 
		args['path_vox1_test'], 
		f'{args["path_vox1_test"]}_noise', 
		args['path_vox1_trials']
	)
	args['num_speaker'] = len(trainer.vox1.train_speakers)

	# data loader
	loaders = data.get_loaders(args, trainer.vox1)
	trainer.train_set = loaders[0]
	trainer.train_loader = loaders[1]
	trainer.enrollment_set = loaders[2]
	trainer.enrollment_loader = loaders[3]

	# model	
	model = SE_ResZNet(args).to(args['device'])
	model.apply(init_weights)

	if args['flag_parent']:
		result, nb_params = summary_string(model, (64,256), device=args['device'])
		trainer.logger.log_text('model_info', result)
		trainer.logger.log_text('nb_params', str(nb_params[1]))
		args['nb_params'] = str(nb_params[1])
		trainer.logger.log_parameter(args)
		
	#XXX
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model = nn.parallel.DistributedDataParallel(model, device_ids=[args['device']], find_unused_parameters=True)
	trainer.model = model

	trainer.spec = LogMelspectrogram(args['winlen'], args['winstep'], args['nfft'], args['samplerate'], args['nfilts'], args['premphasis'], args['winfunc'])

	# criterion
	criterion = {}
	
	classification_loss_function = importlib.import_module('loss.'+ args['classification_loss']).__getattribute__('LossFunction')
	criterion['classification_loss'] = classification_loss_function(args['code_dim'], args['num_speaker']).to(args['device']) 
	criterion_params = list(criterion['classification_loss'].parameters())
	criterion['classification_loss'] = nn.parallel.DistributedDataParallel(criterion['classification_loss'], device_ids=[args['device']], find_unused_parameters=True)
	
	if args['do_train_feature_enhancement']: 
		enhancement_loss = importlib.import_module('loss.'+ args['enhancement_loss']).__getattribute__('LossFunction')
		criterion['enhancement_loss'] = enhancement_loss()

	if args['do_train_code_enhancement']: 
		code_enhancement_loss = importlib.import_module('loss.'+ args['code_enhacement_loss']).__getattribute__('LossFunction')
		criterion['code_enhancement_loss'] = code_enhancement_loss(args['device']).to(args['device']) 
		criterion_params += list(criterion['code_enhancement_loss'].parameters())
		criterion['code_enhancement_loss'] = nn.parallel.DistributedDataParallel(criterion['code_enhancement_loss'], device_ids=[args['device']], find_unused_parameters=True)
	trainer.criterion = criterion

	# optimizer
	if args['optimizer'] == 'adam':
		trainer.optimizer = torch.optim.Adam(
			list(model.parameters()) + criterion_params, 
			lr=args['lr_start'], 
			weight_decay=args['weigth_decay'],
			amsgrad = args['amsgrad']
		)
	elif args['optimizer'] == 'sgd':
		trainer.optimizer = torch.optim.SGD(
			list(model.parameters()) + criterion_params, 
			lr=args['lr_start'], 
			momentum = 0.9,
			weight_decay=args['weigth_decay'],
			nesterov = True
			)

	# lr scheduler
	args['number_iteration'] = len(trainer.train_loader)
	if args['learning_rate_scheduler'] == 'cosine':
		trainer.lr_scheduler = CosineAnnealingWarmRestarts(
			trainer.optimizer, 
			T_0=args['number_iteration'] * args['number_cycle'], 
			eta_min=args['lr_end']
		)
	elif args['learning_rate_scheduler'] == 'step':
		trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
			trainer.optimizer, 10, gamma=0.95)

	# train
	trainer.run()
		
	if args['flag_parent']:	trainer.logger.finish()


if __name__ == '__main__':
	# get arguments
	args, system_args, experiment_args = arguments.get_args()
	
	# set reproducible
	random.seed(args['rand_seed'])
	np.random.seed(args['rand_seed'])
	torch.manual_seed(args['rand_seed'])

	# set gpu device
	if args['usable_gpu'] is None: 
		args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
	else:
		args['gpu_ids'] = args['usable_gpu'].split(',')
	
	if len(args['gpu_ids']) == 0:
		raise Exception('Only GPU env are supported')

	# set DDP
	args['world_size'] = len(args['gpu_ids'])
	args['batch_size'] = args['batch_size'] // (args['world_size'] * args['nb_utt_per_spk'])
	args['num_workers'] = args['num_workers'] // args['world_size']
	
	# check dataset
	data_preprocessor = DataPreprocessor(args['path_musan'], args['path_vox1_test'])
	data_preprocessor.check_environment()
	
	# start
	torch.multiprocessing.set_sharing_strategy('file_system')
	torch.multiprocessing.spawn(
		run, 
		nprocs=args['world_size'], 
		args=(args, experiment_args,)
	)