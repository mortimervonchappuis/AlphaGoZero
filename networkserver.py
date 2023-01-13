from tqdm import tqdm
from os import mkdir
from utilities import TimeLogger
from email_notification import notify
import tensorflow as tf
import numpy as np



class NetworkServer:
	"""Server for ANN evaluation and training"""
	TIMER = TimeLogger()


	def __init__(self, 
				 model_class, 
				 epochs=None, 
				 batch_size=None, 
				 checkpoint=None, 
				 ratio=None, 
				 replay_buffer=None, 
				 weight_path=None,
				 n_moves=None, 
				 n_tree_nodes=None, 
				 n_threads=None, 
				 alpha_net_path=None,
				 beta_net_path=None):
		self.model_class = model_class
		self.epochs = epochs
		self.batch_size = batch_size
		self.checkpoint = checkpoint
		self.ratio = ratio
		self.replay_buffer = replay_buffer
		self.weight_path = weight_path
		self.alpha_net_path = alpha_net_path
		self.beta_net_path = beta_net_path
		self.n_moves = n_moves
		self.n_tree_nodes = n_tree_nodes
		self.n_threads = n_threads
		try:
			if self.weight_path:
				mkdir(self.weight_path)
				print(f'DIRECTORY {self.weight_path} HAS BEEN INITIALIZED.')
		except FileExistsError:
			print(f'DIRECTORY {self.weight_path} ALREADY EXISTS.')


	@TIMER.log
	def eval(self, link, model=None):
		model = model or self.model
		data = np.concatenate(link.recv())
		predictions = model.predict(data)
		link.send(predictions)


	@TIMER.log
	def train(self, batch):
		Xs, PIs, Zs = batch
		return self.model.train_on_batch(x=Xs, y=[PIs, Zs], return_dict=True)
	

	def fill(self, link):
		with tqdm(total=self.replay_buffer.size) as pbar:
			counter = 0
			self.model = self.model_class()
			if self.alpha_net_path:
				self.model.load_weights(self.alpha_net_path)
			while not self.replay_buffer:
				self.eval(link)
				if counter != len(self.replay_buffer):
					pbar.update(len(self.replay_buffer) - counter)
					counter = len(self.replay_buffer)
			if not self.alpha_net_path:
				self.model.save('RB_filler_model.pd')


	def run(self, link):
		with tf.device('GPU'):
			self.model = self.model_class()
			if self.alpha_net_path:
				self.model.load_weights(self.alpha_net_path)
			print('NETWORK SERVER HAS BEEN STARTED!')
			
			# EVALUATION UNTIL REPLAYBUFFER CONTAINS DATA
			self.eval(link) # one call for logging
			self.model.summary()
			while len(self.replay_buffer) < self.batch_size:
				self.eval(link)
			
			# FIRST TRAINING TO MEASURE TIMING
			batch = self.replay_buffer.fetch(self.batch_size)
			history = self.train(batch)
			print('NETWORK SERVER HAS BEEN INITIALIZED!')
			with tqdm(total=self.epochs) as pbar:
				for epoch in range(1, self.epochs + 1):

					# RECALIBRATING NUMBER OF CYCLES
					measured_ratio = self.TIMER['eval'] / self.TIMER['train']
					self.cycles = int(self.ratio / measured_ratio)

					# ALTERNATE TIMING CONTROLL
					self.cycles = self.ratio

					# PROGRESS
					progress = 1/(self.cycles + 1)
					
					# UPDATE STATISTICS
					total_loss = round(history['loss'], 4)
					policy_loss = round(history['output_1_loss'], 4)
					value_loss = round(history['output_2_loss'], 4)
					postfix = {'cycles': f"{self.cycles}", 
							   'loss': f"T: {total_loss} π: {policy_loss} V: {value_loss}", 
							   'RB': f"{len(self.replay_buffer)}"}
					pbar.set_postfix(**postfix)

					# EVALUATION
					pbar.set_description('Evaluating')
					for cycle in range(self.cycles):
						self.eval(link)
						pbar.update(progress)
				
					# TRAINING
					pbar.set_description('Training')
					batch = self.replay_buffer.fetch(self.batch_size)
					history = self.train(batch)
					pbar.update(progress)
				
					# CHECKPOINT
					if self.checkpoint is not None and epoch % self.checkpoint == 0:
						padded_epoch = str(epoch).zfill(len(str(epoch)))
						self.model.save(f'{self.weight_path}/epoch_{padded_epoch}.pd')
					
				# FINALLY
				self.model.save(f'{self.weight_path}/final.pd')
				self.replay_buffer.save()
				notify('\n'.join(f"{key}={val}" for key, val in postfix.items()))


	def serve(self, link, phase):
		with tf.device('GPU'):
			if self.alpha_net_path is None or isinstance(self.alpha_net_path, str):
				self.alpha_model = self.model_class()
				self.beta_model = self.model_class()
				self.alpha_model.load_weights(self.alpha_net_path)
				if self.beta_net_path:
					self.beta_model.load_weights(self.beta_net_path)
			else:
				self.alpha_model = self.alpha_net_path()
				self.beta_model = self.beta_net_path()
			if phase == 'alpha':
				with tqdm(total=self.n_moves) as pbar:
					pbar.set_description('Phase Alpha')
					for _ in range(self.n_moves//2):
						for _ in range(self.n_tree_nodes//self.n_threads):
							self.eval(link, self.alpha_model)
						pbar.update(1)
						for _ in range(self.n_tree_nodes//self.n_threads):
							self.eval(link, self.beta_model)
						pbar.update(1)
			elif phase == 'beta':
				with tqdm(total=self.n_moves) as pbar:
					pbar.set_description('Phase Beta')
					for _ in range(self.n_moves//2):
						for _ in range(self.n_tree_nodes//self.n_threads):
							self.eval(link, self.beta_model)
						pbar.update(1)
						for _ in range(self.n_tree_nodes//self.n_threads):
							self.eval(link, self.alpha_model)
						pbar.update(1)
