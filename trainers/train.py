from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

from utils.ddp_util import all_gather
import utils.metric as metric

class ModelTrainer:
	args = None
	vox1 = None
	model = None
	logger = None
	criterion = None
	optimizer = None
	lr_scheduler = None
	train_set = None
	train_loader = None
	enrollment_set = None
	enrollment_loader = None
	spec = None

	def run(self):
		self.best_eer = 1000
		self.do_test_noise = [0]
		
		for epoch in range(self.args['epoch']):
			self.train(epoch)
			self.test(epoch)
			
	def train(self, epoch):
		self.model.train()
		idx_ct_start = len(self.train_loader)*(int(epoch))

		_loss = 0.
		_loss_clf = 0.
		if self.args['do_train_feature_enhancement']: _loss_fea_enh = 0.
		if self.args['do_train_code_enhancement']: _loss_code_enh = 0.

		with tqdm(total = len(self.train_loader), ncols = 150) as pbar:
			for idx, (m_batch_clean, m_batch_noise, m_batch_referance, m_label) in enumerate(self.train_loader):	#torch.Size([(clean,noise)2, bs, (channel)1, (bins)64, (frames) 256]), torch.Size([bs])
				loss = 0
				self.optimizer.zero_grad()
			
				m_label = m_label.tile(2).to(self.args['device'])
				
				m_batch_clean = self.spec(m_batch_clean.to(self.args['device'], non_blocking=True))
				m_batch_noise = self.spec(m_batch_noise.to(self.args['device'], non_blocking=True))
				m_batch = torch.cat((m_batch_clean, m_batch_noise))

				m_batch_referance = self.spec(m_batch_referance.to(self.args['device'], non_blocking=True))
				m_batch_referance = torch.cat((m_batch_clean, m_batch_referance)).unsqueeze(dim=1)
				
				code, output = self.model(m_batch)
				
				description = '%s epoch: %d '%(self.args['name'], epoch)
				
				# code classification loss
				loss_clf = self.criterion['classification_loss'](code, m_label)
				loss += self.args['weight_classification_loss'] * loss_clf
				_loss_clf += loss_clf.cpu().detach() 
				description += 'loss_clf:%.3f '%(loss_clf)
				
				# feature enhancement loss
				if self.args['do_train_feature_enhancement']:
					loss_fea_enh = self.criterion['enhancement_loss'](output, m_batch_referance)
					loss += self.args['weight_feature_enhancement_loss'] * loss_fea_enh
					_loss_fea_enh += loss_fea_enh.cpu().detach() 
					description += 'loss_fea_enh:%.3f '%(loss_fea_enh)

				# code enhancement loss
				if self.args['do_train_code_enhancement']:
					code_clean = code[:len(code)//2]
					code_noisy = code[len(code)//2:]	
					loss_code_enh = self.criterion['code_enhancement_loss'](code_clean, code_noisy)
					loss += self.args['weight_code_enhancement_loss'] * loss_code_enh
					_loss_code_enh += loss_code_enh.cpu().detach() 
					description += 'loss_code_enh:%.3f '%(loss_code_enh)

				loss.backward()

				_loss += loss.cpu().detach()
				self.optimizer.step()

				description += 'TOT: %.4f'%(loss)
				pbar.set_description(description)
				pbar.update(1)

				# if the current epoch is match to the logging condition, log
				if idx % self.args['number_iteration_for_log'] == 0:
					if idx != 0:
						_loss /= self.args['number_iteration_for_log']
						_loss_clf /= self.args['number_iteration_for_log']
						if self.args['do_train_feature_enhancement']: _loss_fea_enh /= self.args['number_iteration_for_log']
						if self.args['do_train_code_enhancement']: _loss_code_enh /= self.args['number_iteration_for_log']
					
						for p_group in self.optimizer.param_groups:
							lr = p_group['lr']
							break

						if self.args['flag_parent']:
							self.logger.log_metric('loss', _loss, step = idx_ct_start+idx)
							self.logger.log_metric('loss_clf', _loss_clf, step = idx_ct_start+idx)
							self.logger.log_metric('lr', lr, step = idx_ct_start+idx)

							_loss = 0.
							_loss_clf = 0.
							if self.args['do_train_feature_enhancement']:
								self.logger.log_metric('loss_fea_enh', _loss_fea_enh, step = idx_ct_start+idx)
								_loss_fea_enh = 0.
							if self.args['do_train_code_enhancement']:
								self.logger.log_metric('loss_code_enh', _loss_code_enh, step = idx_ct_start+idx)
								_loss_code_enh = 0.

				if self.args['learning_rate_scheduler'] == 'cosine': 
					self.lr_scheduler.step()		
		if self.args['learning_rate_scheduler'] == 'step': 
			self.lr_scheduler.step()		

	def test(self, epoch):
		# clean test data
		self.enrollment_set.Key = 'clean'
		self.embeddings = self._enrollment()
		if self.args['flag_parent']:
			self.cur_eer, min_dcf = self._calculate_eer()
			self.logger.log_metric('EER_clean', self.cur_eer, epoch_step=epoch)
			self.logger.log_metric('Min_DCF_clean', min_dcf, epoch_step=epoch)
			
			if self.cur_eer < self.best_eer:
				self.best_eer = self.cur_eer
				self.logger.log_metric('BestEER_clean', self.best_eer, epoch_step=epoch)
				self.logger.save_model('BestModel_{}_{}'.format(epoch, self.best_eer), self.model.module.state_dict())
				self.do_test_noise = [1]

		self._synchronize()
		self.do_test_noise = all_gather(self.do_test_noise)

		if sum(self.do_test_noise) and epoch >= 100:
			# noise test data
			self.test_noise_set(epoch, 'noise_0')
			self.test_noise_set(epoch, 'noise_5')
			self.test_noise_set(epoch, 'noise_10')
			self.test_noise_set(epoch, 'noise_15')
			self.test_noise_set(epoch, 'noise_20')
			
			self.test_noise_set(epoch, 'speech_0')
			self.test_noise_set(epoch, 'speech_5')
			self.test_noise_set(epoch, 'speech_10')
			self.test_noise_set(epoch, 'speech_15')
			self.test_noise_set(epoch, 'speech_20')
			
			self.test_noise_set(epoch, 'music_0')
			self.test_noise_set(epoch, 'music_5')
			self.test_noise_set(epoch, 'music_10')
			self.test_noise_set(epoch, 'music_15')
			self.test_noise_set(epoch, 'music_20')
			
		self.do_test_noise = [0]

	def test_noise_set(self, epoch, key):
		self.enrollment_set.Key = key
		self.embeddings = self._enrollment()
		if self.args['flag_parent']:
			eer, min_dcf = self._calculate_eer()
			self.logger.log_metric(f'EER_{key}', eer, epoch_step=epoch)
			self.logger.log_metric(f'Min_DCF_{key}', min_dcf, epoch_step=epoch)
		self._synchronize()

	def _enrollment(self):
		"""Return embedding dictionary
		(self.enrollment_set is used for processing)
		"""
		self.model.eval()

		keys = []
		embeddings = []
		
		with torch.set_grad_enabled(False):
			with self.model.no_sync():
				for utt, key in tqdm(self.enrollment_loader, desc='enrollment', ncols=self.args['tqdm_ncols']):
					utt = utt.to(self.args['device'], non_blocking=True).squeeze(0)
					
					utt = self.spec(utt)
					
					keys.extend(key)
					embeddings.append(self.model(utt, only_code = True).to('cpu'))
		
		self._synchronize()
		
		keys = all_gather(keys)
		embeddings = all_gather(embeddings)
		
		embedding_dict = {}
		for i in range(len(keys)):
			embedding_dict[keys[i]] = embeddings[i]
		
		return embedding_dict
	
	def _calculate_eer(self):
		# test
		labels = []
		cos_sims = []

		for item in tqdm(self.vox1.test_trials, desc='test', ncols=self.args['tqdm_ncols']):
			cos_sims.append(self._calculate_cosine_similarity(self.embeddings[item.key1], self.embeddings[item.key2]))
			labels.append(int(item.label))

		eer = metric.calculate_EER(
			scores=cos_sims, labels=labels
		)
		min_dcf = metric.calculate_MinDCF(
			scores=cos_sims, labels=labels
		)
		return eer, min_dcf

	def _synchronize(self):
		torch.cuda.empty_cache()
		dist.barrier()

	def _calculate_cosine_similarity(self, trials, enrollments):
	
		buffer1 = F.normalize(trials, p=2, dim=1)
		buffer2 = F.normalize(enrollments, p=2, dim=1)

		dist = F.pairwise_distance(buffer1.unsqueeze(-1), buffer2.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy()

		score = -1 * np.mean(dist)

		return score
