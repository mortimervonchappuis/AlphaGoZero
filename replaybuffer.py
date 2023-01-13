import numpy as np
from pickle import load, dump
from os.path import isfile
from dihedral import DihedralReflector



class ReplayBuffer:
	def __init__(self, 
				 size, 
				 manager, 
				 reinit=True, 
				 file_name='RPDB.pkl'):
		self.size = size
		self.manager = manager
		self.file_name = file_name
		self.reflector = DihedralReflector()
		if reinit:
			print('REINITIALIZED REPLAY BUFFER')
			self.DB = self.manager.list()
		else:
			if isfile(file_name):
				with open(file_name, 'rb') as file_input:
					self.DB = self.manager.list(load(file_input))
				print('RELOADED REPLAY BUFFER')
			else:
				raise FileNotFoundError(f'{file_name} does not exsist')


	def __len__(self):
		return len(self.DB)


	def __getitem__(self, key):
		return self.DB[key]


	def __setitem__(self, key, value):
		self.DB[key] = value


	def extend(self, data):
		self.DB.extend(data)
		del self.DB[:-self.size]
	

	def save(self):
		with open(self.file_name, 'wb') as file_output:
			data = list(self.DB)
			dump(data, file_output)


	def fetch(self, batch_size):
		Xs, PIs, Zs = [], [], []
		for i in np.random.choice(len(self), batch_size):
			X, PI, Z = self[i]
			R, X, PI = self.reflector(X, PI)
			Xs.append(X)
			PIs.append(PI)
			Zs.append(Z)
		return np.array(Xs), np.array(PIs), np.array(Zs)


	def __bool__(self):
		return len(self) >= self.size



if __name__ == '__main__':
	from multiprocessing import Manager
	m = Manager()
	RP = ReplayBuffer(size=10, manager=m)
	for i in range(20):
		RP.extend([(np.zeros((2,)), np.ones((1,)), i)])
	RP.save()

	RP = ReplayBuffer(size=10, manager=m, reinit=False)
	print(RP.fetch(batch_size=4))