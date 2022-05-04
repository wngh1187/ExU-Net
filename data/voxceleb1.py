import os
from dataclasses import dataclass

@dataclass
class TrainItem:
	path: str
	speaker: str

@dataclass
class EnrollmentItem:
	key: str
	path: str

@dataclass
class TestTrial:
	key1: str
	key2: str
	label: str

class VoxCeleb1:
	@property
	def train_set(self):
		return self.__train_set

	@property
	def train_speakers(self):
		return self.__train_speakers

	@property
	def enrollment_set(self):
		return self.__enrollment_set

	@property
	def test_trials(self):
		return self.__test_trials

	def __init__(self, path_train, path_test, path_test_noise, path_trial):
		# train_set
		self.__train_set = []
		for root, _, files in os.walk(path_train):
			for file in files:
				if '.wav' in file:
					temp = os.path.join(root, file)
					self.__train_set.append(
						TrainItem(
							path=temp,
							speaker=temp.split('/')[-3]
						)
					)

		# train_speakers
		temp = {}
		for item in self.train_set:
			try:
				temp[item.speaker]
			except:
				temp[item.speaker] = None
		self.__train_speakers = temp.keys()

		# enrollment_set
		self.__enrollment_set = {
			'clean': [],
			'noise_0': [],
			'noise_5': [],
			'noise_10': [],
			'noise_15': [],
			'noise_20': [],
			'speech_0': [],
			'speech_5': [],
			'speech_10': [],
			'speech_15': [],
			'speech_20': [],
			'music_0': [],
			'music_5': [],
			'music_10': [],
			'music_15': [],
			'music_20': []
		}
		self._parse_enrollment(path_test, 'clean')
		self._parse_enrollment(f'{path_test_noise}/noise_0', 'noise_0')
		self._parse_enrollment(f'{path_test_noise}/noise_5', 'noise_5')
		self._parse_enrollment(f'{path_test_noise}/noise_10', 'noise_10')
		self._parse_enrollment(f'{path_test_noise}/noise_15', 'noise_15')
		self._parse_enrollment(f'{path_test_noise}/noise_20', 'noise_20')

		self._parse_enrollment(f'{path_test_noise}/speech_0', 'speech_0')
		self._parse_enrollment(f'{path_test_noise}/speech_5', 'speech_5')
		self._parse_enrollment(f'{path_test_noise}/speech_10', 'speech_10')
		self._parse_enrollment(f'{path_test_noise}/speech_15', 'speech_15')
		self._parse_enrollment(f'{path_test_noise}/speech_20', 'speech_20')

		self._parse_enrollment(f'{path_test_noise}/music_0', 'music_0')
		self._parse_enrollment(f'{path_test_noise}/music_5', 'music_5')
		self._parse_enrollment(f'{path_test_noise}/music_10', 'music_10')
		self._parse_enrollment(f'{path_test_noise}/music_15', 'music_15')
		self._parse_enrollment(f'{path_test_noise}/music_20', 'music_20')

		# test_trials
		self.__test_trials = self._parse_trials(path_trial)
		
		# error check
		assert len(self.train_set) == 148642, f'len(train_set): {len(self.train_set)}'
		assert len(self.train_speakers) == 1211, f'len(train_speakers): {len(self.train_speakers)}'
		assert len(self.enrollment_set["clean"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["clean"])}'
		assert len(self.enrollment_set["noise_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_0"])}'
		assert len(self.enrollment_set["noise_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_5"])}'
		assert len(self.enrollment_set["noise_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_10"])}'
		assert len(self.enrollment_set["noise_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_15"])}'
		assert len(self.enrollment_set["noise_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["noise_20"])}'
		assert len(self.enrollment_set["speech_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_0"])}'
		assert len(self.enrollment_set["speech_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_5"])}'
		assert len(self.enrollment_set["speech_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_10"])}'
		assert len(self.enrollment_set["speech_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_15"])}'
		assert len(self.enrollment_set["speech_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["speech_20"])}'
		assert len(self.enrollment_set["music_0"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_0"])}'
		assert len(self.enrollment_set["music_5"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_5"])}'
		assert len(self.enrollment_set["music_10"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_10"])}'
		assert len(self.enrollment_set["music_15"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_15"])}'
		assert len(self.enrollment_set["music_20"]) == 4874, f'len(enrollment_set): {len(self.enrollment_set["music_20"])}'
		assert len(self.test_trials) == 37720, f'len(test_trials): {len(self.test_trials)}'

	def _parse_enrollment(self, path, key):	
		for root, _, files in os.walk(path):
			for file in files:
				if '.wav' in file:
					temp = os.path.join(root, file)
					self.__enrollment_set[key].append(
						EnrollmentItem(
							path=temp,
							key='/'.join(temp.split('/')[-3:])
						)
					)

	def _parse_trials(self, path):
		trials = []

		f = open(path) 
		for line in f.readlines():
			strI = line.split(' ')
			trials.append(
				TestTrial(
					key1=strI[1].replace('\n', ''),
					key2=strI[2].replace('\n', ''),
					label=strI[0]
				)
			)
		
		return trials
