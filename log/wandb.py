import os
import torch
import wandb
from wandb import AlertLevel
from .interface import ExperimentLogger

class WandbLogger(ExperimentLogger):
	"""Save experiment logs to wandb
	"""
	def __init__(self, path, name, group, project, entity, tags, script=None, save_dir = None):
		self.run = wandb.init(
				group=group,
				project=project,
				entity=entity,
				tags=tags
			)
		wandb.run.name = name
		path = os.path.join("/", path, project, "/".join(tags), name)
		self.paths = {
			'model' : f'{path}/model',
		}
		# upload zip file
		wandb.save(save_dir + "/script/script.zip")


	def log_metric(self, name, value, step=None, epoch_step=None):
		if epoch_step is not None:
			wandb.log({
				name: value,
				'epoch_step': epoch_step})
		wandb.log({name: value}, step = step)   

	def log_text(self, name, text):
		pass

	def log_image(self, name, image):
		wandb.log({name: [wandb.Image(image)]})

	def log_parameter(self, dictionary):
		wandb.config.update(dictionary)

	def save_model(self, name, state_dict):
		path = f'{self.paths["model"]}/BestModel.pt'
		wandb.save(path)

	def finish(self):
		wandb.finish()