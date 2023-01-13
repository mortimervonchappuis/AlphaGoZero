from interface import Interface
from networkserver import NetworkServer
from emulation import Emulation
from replaybuffer import ReplayBuffer
from multiprocessing import Manager, Process
from pickle import dump
from tqdm import tqdm



class AlphaZero:
	def __init__(self, 
				 model, 
				 MDP, 
				 n_moves, 
				 n_tree_nodes, 
				 n_threads, 
				 n_sampling_moves, 
				 capacity):
		"""Returns an AlphaZero instance

		MDP:				environment class i.e. a game
		model:				learning model the predicts the policy
							and the value for a state
		capacity:			an integer that specifies the size of the storage
		n_moves:			an integer that specifies the maximal number of moves
							before the game is terminated
		n_tree_nodes:		an integer number of simulations that specifies how
							many simulations are performed on the MCTS per action
		n_sampling_moves:	number of moves that are drawn from PI, the remaining 
							moves will be argmax of PI 
		"""
		self.model = model
		self.MDP = MDP
		self.capacity = capacity
		self.n_moves = n_moves
		self.n_tree_nodes = n_tree_nodes
		self.n_threads = n_threads
		self.n_sampling_moves = n_sampling_moves


	def __call__(self, 
				 epochs, 
				 checkpoint, 
				 batch_size, 
				 n_processes, 
				 ratio, 
				 weight_path,
				 net_path=None,
				 reinit=True):
		with Manager() as manager:
			replay_buffer = ReplayBuffer(size=self.capacity, manager=manager, reinit=reinit)
	
			NET_CONFIG = {'model_class': self.model, 
						  'epochs': epochs, 
						  'batch_size': batch_size, 
						  'checkpoint': checkpoint, 
						  'ratio': ratio, 
						  'replay_buffer': replay_buffer, 
						  'weight_path': weight_path,
						  'alpha_net_path': net_path}
			EMU_CONFIG = {'replay_buffer': replay_buffer, 
						  'MDP_class': self.MDP, 
						  'n_tree_nodes': self.n_tree_nodes, 
						  'n_sampling_moves': self.n_sampling_moves, 
						  'n_threads': self.n_threads, 
						  'n_moves': self.n_moves}
	
			network_server = lambda link: NetworkServer(**NET_CONFIG).run(link)
			emulation_server = lambda link, seed: Emulation(**EMU_CONFIG).run(link, seed)
			emergency = None # lambda: replay_buffer.save()
			interface = Interface(network_server, emulation_server, emergency)
			with interface:
				interface.serve(n_processes)


	def eval(self,
			 n_processes,
			 alpha_net_path,
			 beta_net_path):
		with Manager() as manager:
			NET_CONFIG = {'model_class': self.model, 
						  'alpha_net_path': alpha_net_path,
						  'beta_net_path': beta_net_path,
						  'n_moves': self.n_moves, 
						  'n_tree_nodes': self.n_tree_nodes, 
						  'n_threads': self.n_threads}
			EMU_CONFIG = {'MDP_class': self.MDP, 
						  'n_tree_nodes': self.n_tree_nodes, 
						  'n_sampling_moves': self.n_sampling_moves//2, 
						  'n_threads': self.n_threads, 
						  'n_moves': self.n_moves}

			network_server = lambda link, phase: NetworkServer(**NET_CONFIG).serve(link, phase)
			emulation_server = lambda link, seed, score, phase: Emulation(**EMU_CONFIG).eval(link, seed, score, phase)
			score = manager.list()
			with Interface(network_server, emulation_server) as interface:
				result = interface.eval(n_processes, score)
			print(f"RESULT {round((result + 1) * 50, 1)}%")


	def fill(self, 
			 n_processes,
			 reinit, 
			 net_path):
		with Manager() as manager:
			replay_buffer = ReplayBuffer(size=self.capacity, manager=manager, reinit=reinit)
	
			NET_CONFIG = {'model_class': self.model, 
						  'replay_buffer': replay_buffer, 
						  'alpha_net_path': net_path}
			EMU_CONFIG = {'replay_buffer': replay_buffer, 
						  'MDP_class': self.MDP, 
						  'n_tree_nodes': self.n_tree_nodes, 
						  'n_sampling_moves': self.n_sampling_moves, 
						  'n_threads': self.n_threads, 
						  'n_moves': self.n_moves, 
						  'temperatur': 1}

			network_server = lambda link: NetworkServer(**NET_CONFIG).fill(link)
			emulation_server = lambda link, seed: Emulation(**EMU_CONFIG).run(link, seed)
			with Interface(network_server, emulation_server) as interface:
				interface.serve(n_processes)
			replay_buffer.save()


	def train(self, 
			  epochs, 
			  batch_size, 
			  net_path=None):

		def server():
			import tensorflow as tf
	
			with tf.device('GPU'):
				with Manager() as manager:
					replaybuffer = ReplayBuffer(size=self.capacity, manager=manager, reinit=False)
					model = self.model()
					if net_path:
						model.load_weights(net_path)
					with tqdm(total=epochs) as pbar:
						for epoch in range(epochs):
							Xs, PIs, Zs = replaybuffer.fetch(batch_size)
							history = model.train_on_batch(x=Xs, y=[PIs, Zs], return_dict=True)
							pbar.update(1)
							pbar.set_postfix(**history)
					model.predict(np.zeros((1, 9, 9, 3)))
					model.save('trained_model.pd')

		process = Process(target=server)
		process.start()
		process.join()





if __name__ == '__main__':
	from model import ResNet
	from go import Go
	from utilities import factory
	from os import system
	import numpy as np
	from email_notification import notify


	MDP = factory(Go, grid=(9, 9), komi=5.5)
	#MDP = factory(Go, grid=(6, 6), komi=2.5)

	def extend(model):
		model.load_weights('6x6/12/trained_model.pd')
		#model.build((None, 8, 8, 3))
		#return model
		#model.summary()
		#model.save('extended.pd')
		model.extend(n_value_units=128, 
					 output_shape=(9, 9, 3), 
					 n_additional_blocks=3)
		model.reduce()
		model.build(input_shape=(None,) + MDP().X.shape)
		return model
	

	#Model = factory(ResNet,
	#				func=extend, 
	#				n_kernels=256, 
	#				n_residual_blocks=6, 
	#				n_value_units=64) # 256

	#Model = factory(ResNet,
	#				n_kernels=256, 
	#				n_residual_blocks=6, 
	#				n_value_units=64)

	def lock(model):
		#model.load_weights('9x9_long/final.pd')
		model.reduce()
		#model.lock(0)
		return model

	Model = factory(ResNet,
					n_kernels=256, 
					func=lock, 
					n_residual_blocks=9, 
					learning_rate=0.0001,
					n_value_units=128)




	def lock_long(model):
		#model.reduce()
		model.load_weights('9x9_long/final.pd')
		#model.reduce()
		#model.load_weights('9x9_solo/model_16.pd')
		#model.lock(0)
		return model

	Model_long = factory(ResNet,
					n_kernels=256, 
					func=lock_long, 
					n_residual_blocks=9, 
					learning_rate=0.0001,
					n_value_units=128)

	def lock_short(model):
		#model.reduce()
		model.load_weights('9x9_short/final.pd')
		#model.lock(0)
		return model

	Model_short = factory(ResNet,
					n_kernels=256, 
					func=lock_short, 
					n_residual_blocks=9, 
					learning_rate=0.0001,
					n_value_units=128)

	def lock_solo(model):
		model.reduce()
		model.load_weights('9x9_solo/final.pd')
		#model.load_weights('trained_model.pd')

		#model.reduce()
		#model.lock(0)
		return model

	Model_solo = factory(ResNet,
					n_kernels=256, 
					func=lock_solo, 
					n_residual_blocks=9, 
					learning_rate=0.0001,
					n_value_units=128)

	#model = Model()
	#model.load_weights('extended.pd')
	#model.build(input_shape=(None,) + MDP().X.shape)
	#print(model.predict(np.zeros((1, 9, 9, 3))))
	#model.summary()
	#model.save('extended.pd')
	#exit()

	AZ = AlphaZero(MDP=MDP, 
				   model=Model_solo, 
				   n_moves=100, 
				   n_tree_nodes=300, 
				   n_threads=8, 
				   n_sampling_moves=12, 
				   capacity=int(1e6))

	#model = Model()
	#model.summary()
	#data = np.stack([MDP().X], axis=0)
	#print(data.shape)
	#print(model.predict(data))

	#net = Model()
	#net.predict(np.zeros((1, 9, 9, 3)))
	#net.load_weights('9x9_solo/epoch_2000.h5')
	#net.save('9x9_solo/model_10.pd')
	#quit()


	#AZ(epochs=2000 * 4, 
	#   checkpoint=2000, #1000, 
	#   batch_size=1024, 
	#   n_processes=128, 
	#   ratio=80, 
	#   weight_path='9x9_solo', 
	#   net_path='9x9_solo/final.pd', 
	#   reinit=False)

	#notify('Started to fill ReplayBuffer')
	#try:
	#AZ.fill(n_processes=128, 
	#		reinit=True, 
	#		net_path='9x9_long/final.pd')
	#except:
	#	notify('ReplayBuffer filling has been terminated')
		#system('shutdown now')
	#	quit()
#	notify('ReplayBuffer has been filled')


	#AZ.train(batch_size=1024, 
	#		 epochs=int(5000),
	#		 net_path='trained_model.pd')

	AZ.eval(n_processes=128, 
			alpha_net_path=Model_long,
			beta_net_path=Model_solo)#'7x7/model_0.pd')#'6x6/2/trained_model.pd')
	quit()


	"""
	FINAL MODELS
	long  vs solo  85.3%
	long  vs short 84.7%
	solo  vs short 71.2%
	long  vs R    100.0%
	short vs R    100.0%
	solo  vs R     99.4%
	"""

	#### 9x9 solo
	# 11 vs 10 = 56.0%
	#  9 vs  8 =  5.1%
	# 11 vs  8 = 14.0%
	# 11 vs  7 = 11.5%
	#  8 vs  7 = 65.0ss%
	#############

	#train = lambda: AZ.train(batch_size=4096, 
	#						 epochs=int(1e4),
	#						 net_path='trained_model.pd')
	#for n in range(7, 13):
	#	system(f'cp 6x6/{n}/RPDB.pkl RPDB.pkl')
	#	AZ.train(batch_size=4096, 
	#			 epochs=int(1e4), 
	#			 net_path='trained_model.pd')
	#	system(f'cp -r trained_model.pd 6x6/{n}/trained_model.pd')

	#exit()

	"""
	ASYNCHRONOUS
	"""
#
#	#try:
#	system(f'cp RPDB.pkl RPDB_copy.pkl')
#	skip = 8
#	komi_schedule = [1.5, 1.5, 1.5, 1.5,  
#					 2.5, 2.5, 3.5, 3.5,  
#					 4.5, 4.5, 5.5, 5.5,  
#					 5.5, 5.5, 5.5, 5.5,  
#					 5.5, 5.5, 5.5, 5.5]
#	lock_schedule = [0] * 44
#	lr_schedule   = [0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.01,   0.01,   0.01,   0.01,  
#					 0.01,   0.01,   0.01,   0.01,   0.001,  0.001,  0.001,  0.001,  
#					 0.001,   0.001,  0.001,  0.001, 0.001,   0.001,  0.001,  0.001, 
#					 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,  0.0001, 0.0001, 0.0001,
#					 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,  0.0001, 0.0001, 0.0001]
#	for i, komi, n_lock, lr in zip(range(skip, 44), 
#								   komi_schedule[skip:], 
#								   lock_schedule[skip:], 
#								   lr_schedule[skip:]):
#		# CREATE MDP
#		MDP = factory(Go, grid=(9, 9), komi=komi)
#		
#		# DEFINE MODEL SETUP
#		def lock(model):
#			model.load_weights('9x9_solo/final.pd')
#			model.lock(n_lock)
#			return model
#		
#		# CREATE MODEL
#		Model = factory(ResNet,
#						n_kernels=256, 
#						func=lock, 
#						n_residual_blocks=8, 
#						learning_rate=lr, 
#						n_value_units=128)
#
#		# CREATE ALPHA ZERO
#		AZ = AlphaZero(MDP=MDP, 
#					   model=Model, 
#					   n_moves=80, 
#					   n_tree_nodes=360, 
#					   n_threads=8, 
#					   n_sampling_moves=12, 
#					   capacity=int(2e6))
#		
#		# START TRAINING
#		AZ(epochs=2000, 
#		   checkpoint=None, #1000, 
#		   batch_size=1024, 
#		   n_processes=128, 
#		   ratio=80, 
#		   weight_path='9x9_solo', 
#		   #net_path='7x7/final_model.pd', 
#		   reinit=False)
#
#		# SAVE CURRENT PARAMETERS
#		system(f'cp -r 9x9_solo/final.pd 9x9_solo/model_{i + 1}.pd')
#	#except:
#	#	print('FAIL')
#	#	notify('System has been terminated')
#	#	#system('shutdown now')
#	#notify('system has finalized')
#	#system('shutdown now')
	
	"""
	SYNCHRONOUS
	"""
	
	#try:

	system(f'cp RPDB.pkl RPDB_copy.pkl')
	skip = 9
	komi_schedule = [1.5, 1.5, 1.5, 1.5,  
					 2.5, 2.5, 3.5, 3.5,  
					 4.5, 4.5, 5.5, 5.5,  
					 5.5, 5.5, 5.5, 5.5,  
					 5.5, 5.5, 5.5, 5.5]
	lock_schedule = [0] * 44
	lr_schedule   = [0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.1,    0.01,   0.01,   0.01,   0.01,  
					 0.01,   0.01,   0.01,   0.01,   0.001,  0.001,  0.001,  0.001,  
					 0.001,   0.001,  0.001,  0.001, 0.001,   0.001,  0.001,  0.001, 
					 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,  0.0001, 0.0001, 0.0001,
					 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,  0.0001, 0.0001, 0.0001]
	for i, komi, n_lock, lr in zip(range(skip, 44), 
								   komi_schedule[skip:], 
								   lock_schedule[skip:], 
								   lr_schedule[skip:]):
		# CREATE MDP
		MDP = factory(Go, grid=(9, 9), komi=komi)
		
		# DEFINE MODEL SETUP
		def lock(model):
			model.load_weights('9x9_solo/final.pd')
			model.lock(n_lock)
			return model
		
		# CREATE MODEL
		Model = factory(ResNet,
						n_kernels=256, 
						func=lock, 
						n_residual_blocks=8, 
						learning_rate=0.0001,#lr, 
						n_value_units=128)

		# CREATE ALPHA ZERO
		AZ = AlphaZero(MDP=MDP, 
					   model=Model, 
					   n_moves=80, 
					   n_tree_nodes=360, 
					   n_threads=8, 
					   n_sampling_moves=12, 
					   capacity=int(5e5))
		
		# START FILLING
		AZ.fill(n_processes=128, 
				reinit=True, 
				net_path='trained_model.pd')

		# START TRAINING
		AZ.train(batch_size=1024, 
				 epochs=int(1e3),
				 net_path='9x9_solo/model_8.pd')#'trained_model.pd')

		# SAVE CURRENT PARAMETERS
		system(f'cp -r trained_model.pd 9x9_solo/model_{i + 1}.pd')
		system(f'cp -r trained_model.pd 9x9_solo/fina_1l.pd')
		#quit()


	# PHASES
	# 6x6 capacity 1000000 - treenodes 240 - 10000 epochs - rounds 12 -           init 16 stunden pro runde
	# 7x7 capacity 1000000 - treenodes 300 -  2000 epochs - rounds 16 - ratio 50  jump start 12 stunden pro runde - 
	# 8x8 capacity 2000000 - treenodes 360 -  2000 epochs - rounds 16 - ratio 80  fill up 40 stunden pro runde 18 stunden
	# 9x9 capacity 2000000 - treenodes 360 -  2000 epochs - rounds 16 - ratio 80  fillup? 65 Stunden 24 Stunden pro Runde


	# MODELS
	# 6x6 n_value_units  64 - n_res_blocks 6
	# 7x7 n_value_units  88 - n_res_blocks 7
	# 8x8 n_value_units  96 - n_res_blocks 8
	# 9x9 n_value_units 128 - n_res_blocks 9




	#'6x6/final.pd'
	#'RB_filler_model.pd'

# TEST GAMES 7x7
#   R v 16 97.5%
#   0 v 16 91.7%
#   1 v 16 85.6%
#   2 v 16 91.5%
#   3 v 16 91.8%
#   4 v 16 89.0%
#   5 v 16 87.1%
#   6 v 16 86.5%
#   7 v 16 82.0%
#   8 v 16 79.0%
#   9 v 16 73.6%
#  10 v 16 69.5%
#  11 v 16 70.7%
#  12 v 16 64.2%
#  13 v 16 57.4%
#  14 v 16 55.6%
#  15 v 16 44.8%


# TRAINING 7x7
# lr=0.01,   komi=0.5, n=1000, ratio=100
# lr=0.001,  komi=1.5, n=1000, ratio=100



# TEST GAMES
#  R v  1 75.3% komi=0.5

#  R v  2 97.7% komi=0.5
#  1 v  2 92.9% komi=0.5

#  R v  3 97.7% komi=1.5
#  1 v  3 93.5% komi=1.5
#  2 v  3 57.2% komi=1.5
#  2 v  3 54.2% komi=2.5

#  R v  4 99.0% komi=2.5
#  1 v  4 88.2% komi=2.5
#  2 v  4 70.0% komi=2.5
#  3 v  4 65.1% komi=2.5

#  R v  5 98.8% komi=2.5
#  1 v  5 97.0% komi=2.5
#  2 v  5 68.7% komi=2.5
#  3 v  5 76.7% komi=2.5
#  4 v  5 64.9% komi=2.5

#  R v  6 96.1% komi=2.5
#  1 v  6 98.1% komi=2.5
#  2 v  6 94.0% komi=2.5
#  3 v  6 87.0% komi=2.5
#  4 v  6 84.7% komi=2.5
#  5 v  6 75.4% komi=2.5

#  R v  7 96.9% komi=2.5
#  1 v  7 98.0% komi=2.5
#  2 v  7 93.7% komi=2.5
#  3 v  7 91.3% komi=2.5
#  4 v  7 94.9% komi=2.5
#  5 v  7 89.5% komi=2.5
#  6 v  7 66.4% komi=2.5

#  R v  8 98.7% komi=2.5
#  1 v  8 98.8% komi=2.5
#  2 v  8 92.1% komi=2.5
#  3 v  8 87.1% komi=2.5
#  4 v  8 90.2% komi=2.5
#  5 v  8 82.7% komi=2.5
#  6 v  8 71.5% komi=2.5
#  7 v  8 63.6% komi=2.5

#  R v  9 98.3% komi=2.5
#  1 v  9 97.4% komi=2.5
#  2 v  9 95.7% komi=2.5
#  3 v  9 88.7% komi=2.5
#  4 v  9 90.0% komi=2.5
#  5 v  9 79.1% komi=2.5
#  6 v  9 72.0% komi=2.5
#  7 v  9 72.6% komi=2.5
#  8 v  9 67.4% komi=2.5

#  R v 10 98.9% komi=2.5
#  1 v 10 97.9% komi=2.5
#  2 v 10 91.5% komi=2.5
#  3 v 10 85.1% komi=2.5
#  4 v 10 87.1% komi=2.5
#  5 v 10 80.3% komi=2.5
#  6 v 10 69.8% komi=2.5
#  7 v 10 61.6% komi=2.5
#  8 v 10 67.6% komi=2.5
#  9 v 10 59.2% komi=2.5

#  R v 11 98.0% komi=2.5
#  1 v 11 95.8% komi=2.5
#  2 v 11 88.3% komi=2.5
#  3 v 11 88.8% komi=2.5
#  4 v 11 86.2% komi=2.5
#  5 v 11 77.0% komi=2.5
#  6 v 11 70.0% komi=2.5
#  7 v 11 50.4% komi=2.5
#  8 v 11 59.5% komi=2.5
#  9 v 11 42.5% komi=2.5
# 10 v 11 47.4% komi=2.5

#  R v 12  100% komi=2.5
#  1 v 12 97.1% komi=2.5
#  2 v 12 96.7% komi=2.5
#  3 v 12 89.3% komi=2.5
#  4 v 12 89.6% komi=2.5
#  5 v 12 79.8% komi=2.5
#  6 v 12 68.2% komi=2.5
#  7 v 12 61.0% komi=2.5
#  8 v 12 60.5% komi=2.5
#  9 v 12 53.5% komi=2.5
# 10 v 12 62.9% komi=2.5
# 11 v 12 61.5% komi=2.5



# 3 is able to poke out dead shape of 5 against 2
# 4 still not totally able t differenciate between true and false eyes


# MODELS
# 1 lr = 0.01
# 2 lr = 0.001
# 3 lr = 0.0001
# 4 lr = 0.0001


# C
# 100 v 10  9.0%
# 10  v 1  37.7%
# 3   v 1  53.3%
# 5   v 1  48.1%