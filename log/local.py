import os
import time
import torch
import shutil
import zipfile
from threading import Thread
from .interface import ExperimentLogger

def zipdir(path, ziph):
	for root, dirs, files in os.walk(path):
		for file in files:
		
			fn, ext = os.path.splitext(file)
			if ext != ".py":
				continue
			
			ap = '/'.join(os.path.abspath(file).split('/')[:-1])
			ziph.write(os.path.join(root, file))

class LocalLogger(ExperimentLogger):
	"""Save experiment logs to local storage.
	Note that logs are synchronized every 10 seconds.
	"""
	def __init__(self, path, name, project, tags, description=None, script=None):
		# set path
		path = os.path.join("/", path, project, "/".join(tags), name)
		self.paths = {
			'metric': f'{path}/metric',
			'text'  : f'{path}/text',
			'image' : f'{path}/image',
			'model' : f'{path}/model',
			'script': f'{path}/script',
		}
		
		# make directory
		# if os.path.exists(path):
		# 	shutil.rmtree(path)
		for dir in self.paths.values():
			if not os.path.exists(dir): os.makedirs(dir) 

		# save description
		if description is not None:
			self.log_text('description', description)

		# zip codes
		with zipfile.ZipFile(
			self.paths['script'] + "/script.zip", "w", zipfile.ZIP_DEFLATED
		) as f_zip:
			zipdir(script, f_zip)

		# synchronize thread
		self._metrics = {}
		thread_synchronize = Thread(target=self._sync)
		thread_synchronize.setDaemon(True)
		thread_synchronize.start()

	def _sync(self):
		"""Flush files
		"""
		while(True):
			for txt in self._metrics.values():
				txt.flush()
			time.sleep(10)

	def log_text(self, name, text):
		path = f'{self.paths["text"]}/{name}.txt'
		mode = 'a' if os.path.exists(path) else 'w'
		file = open(path, mode, encoding='utf-8')
		file.write(text)
		file.close()

	def log_parameter(self, dictionary):
		for k, v in dictionary.items():
			self.log_text('parameters', f'{k}: {v}\n')

	def log_metric(self, name, value, step=None, epoch_step=None):
		if name not in self._metrics.keys():
			path = f'{self.paths["metric"]}/{name}.txt'
			self._metrics[name] = open(path, 'w', encoding='utf-8').close()
			self._metrics[name] = open(path, 'a', encoding='utf-8')

		if step is None:
			self._metrics[name].write(f'{name}: {value}\n')
		else:
			self._metrics[name].write(f'[{step}] {name}: {value}\n')
	
	def log_image(self, name, image):
		path = f'{self.paths["image"]}/{name}.png'
		image.save(path, 'PNG')
	
	def save_model(self, name, state_dict):
		path = f'{self.paths["model"]}/BestModel.pt'
		torch.save(state_dict, path)

		path = f'{self.paths["model"]}/{name}.pt'
		torch.save(state_dict, path)
	  
	def finish(self):
		pass