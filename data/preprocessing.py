import os
import soundfile as sf
from tqdm import tqdm
from scipy.io import wavfile
from .musan import MusanNoise

class DataPreprocessor:
	def __init__(self, path_musan, path_vox1_test):
		self.path_musan = path_musan
		self.path_vox1_test = path_vox1_test

		dir = self._leaf_dir(self.path_musan)
		self.path_musan_split = self.path_musan.replace(dir, f'{dir}_split')

		self.dir_vox1_test = self._leaf_dir(self.path_vox1_test)
		self.path_vox1_test_noise = self.path_vox1_test.replace(
				self.dir_vox1_test, f'{self.dir_vox1_test}_noise')

	def _leaf_dir(self, path):
		"""
		Return leaf sub directory name from path
		"""
		if '.' in path:
			name = os.path.dirname(path).split('/')[-1]
		elif path[-1] == '/':
			name = os.path.dirname(path).split('/')[-1]
		else:
			name = os.path.basename(path)

		return name

	def check_environment(self):
		"""Check if preprocessing has been done
		"""
		if not os.path.exists(self.path_musan_split):
			print('You need to split musan dataset')
			print('Now processing...')
			self.split_musan()
		
		if not os.path.exists(self.path_vox1_test_noise):		
			self.musan = MusanNoise(self.path_musan_split + '/test')
			print('You need to make noisy test set')
			print('Now processing...')
			self.init_noisey_test_set()

	def split_musan(self):
		"""
		Split MUSAN dataset for 
			- to divide train/test set
			- to increase reading speed
		"""
		winlen = 16000 * 5
		winstep = 16000 * 3

		for root, _, files in os.walk(self.path_musan):
			for file in tqdm(files):
				if '.wav' in file:
					num = int(file[-8:-4])
					assert 0 <= num and num < 1000
					category = 'train' if num % 2 == 0 else 'test'

					file = os.path.join(root, file)
					temp = f'{self.path_musan_split}/{category}'
					destination = file.replace(self.path_musan, temp
									 ).replace('.', '')
					os.makedirs(destination, exist_ok=True)

					sr, data = wavfile.read(file)
					num_file = (len(data) - winlen) // winstep

					if 0 < num_file:
						for i in range(num_file):
							wavfile.write(f'{destination}/{i * 3}_{(i * 3) + 5}.wav', sr, data[i * winstep:i * winstep + winlen])
					else:
						wavfile.write(f'{destination}/all.wav', sr, data)

	def init_noisey_test_set(self):
		"""Inject musan noise to vox1 testset for making noisy test set
		"""
		for root, _, files in tqdm(os.walk(self.path_vox1_test)):
			for file in files:
				if '.wav' in file:
					path = os.path.join(root, file)
					self._inject_noise(path, 'noise', 0)
					self._inject_noise(path, 'noise', 5)
					self._inject_noise(path, 'noise', 10)
					self._inject_noise(path, 'noise', 15)
					self._inject_noise(path, 'noise', 20)
					self._inject_noise(path, 'speech', 0)
					self._inject_noise(path, 'speech', 5)
					self._inject_noise(path, 'speech', 10)
					self._inject_noise(path, 'speech', 15)
					self._inject_noise(path, 'speech', 20)
					self._inject_noise(path, 'music', 0)
					self._inject_noise(path, 'music', 5)
					self._inject_noise(path, 'music', 10)
					self._inject_noise(path, 'music', 15)
					self._inject_noise(path, 'music', 20)

	def _inject_noise(self, path, category, snr):
		data, sr = sf.read(path)
		self.musan.Category = [category]
		self.musan.SNR[category] = (snr, snr)
		destination = path.replace(
			self.dir_vox1_test, f'{self.dir_vox1_test}_noise/{category}_{snr}')
		os.makedirs(os.path.dirname(destination), exist_ok=True)
		data = self.musan(data)
		sf.write(destination, data, sr)