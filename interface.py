from multiprocessing import Process, Pipe
from utilities import group



class Interface:
	def __init__(self, server_func, worker_func, emergency=None):
		self.server_func = server_func
		self.worker_func = worker_func
		self.emergency = emergency


	def __enter__(self):
		return self


	def __exit__(self, types, values, traceback):
		if traceback is not None and self.emergency is not None:
			self.emergency()
			print('EMERGENCY FUNCTION HAS BEEN CALLED!')
			self.server.terminate()
		self.server.join()
		for worker in self.workers:
			worker.terminate()
		self.inter.terminate()


	def serve(self, n_processes):
		# Init Worker Connections
		self.workers, self.inter_links = [], []
		for PID in range(n_processes):
			inter_link, worker_link = Link.create()
			worker = Process(target=self.worker_func, args=(worker_link, PID))
			self.workers.append(worker)
			self.inter_links.append(inter_link)

		# Init Server Connections
		inter_out, server_in = Pipe()
		server_out, inter_in = Pipe()
		self.inter_link = Link(inter_in, inter_out)
		self.server_link = Link(server_in, server_out)
		self.inter  = Process(target=self.wait, args=(self.inter_link, self.inter_links))
		self.server = Process(target=self.server_func, args=(self.server_link,))
		self.inter.start()
		self.server.start()
		for worker in self.workers:
			worker.start()
		self.server.join()

		# CLEAN UP
		self.inter.terminate()
		for worker in self.workers:
			worker.terminate()


	def eval(self, n_processes, score):

		# PHASE ALPHA

		# INIT WORKER CONNECTIONS
		self.workers, self.inter_links = [], []
		for PID in range(n_processes):
			inter_link, worker_link = Link.create()
			worker = Process(target=self.worker_func, args=(worker_link, PID, score, 'alpha'))
			self.workers.append(worker)
			self.inter_links.append(inter_link)

		# INIT SERVER CONNECTIONS
		inter_out, server_in = Pipe()
		server_out, inter_in = Pipe()
		self.inter_link  = Link(inter_in, inter_out)
		self.server_link = Link(server_in, server_out)
		
		self.inter  = Process(target=self.wait, args=(self.inter_link, self.inter_links))
		self.server = Process(target=self.server_func, args=(self.server_link, 'alpha'))
		self.inter.start()
		self.server.start()
		for worker in self.workers:
			worker.start()
		self.server.join()
		for worker in self.workers:
			worker.terminate()
		result = sum(score)/len(score)
		del score[:]
		self.inter.terminate()

		# PHASE BETA

		# INIT WORKER CONNECTIONS
		self.workers, self.inter_links = [], []
		for PID in range(n_processes):
			inter_link, worker_link = Link.create()
			worker = Process(target=self.worker_func, args=(worker_link, PID, score, 'beta'))
			self.workers.append(worker)
			self.inter_links.append(inter_link)

		# INIT SERVER CONNECTIONS
		inter_out, server_in = Pipe()
		server_out, inter_in = Pipe()
		self.inter_link  = Link(inter_in, inter_out)
		self.server_link = Link(server_in, server_out)

		self.inter  = Process(target=self.wait, args=(self.inter_link, self.inter_links))
		self.server = Process(target=self.server_func, args=(self.server_link, 'beta'))
		self.inter.start()
		self.server.start()
		for worker in self.workers:
			worker.start()
		self.server.join()
		result -= sum(score)/len(score)
		return result/2


	@staticmethod
	def wait(mono_link, multi_links):
		while True:
			data = [link.recv() for link in multi_links]
			Ps, Vs = mono_link(data)
			Ps, Vs = group(len(multi_links), Ps, Vs)
			for link, P, V in zip(multi_links, Ps, Vs):
				link.send([P, V])



class Link:
	"""Bidirectional Pipe"""
	def __init__(self, inputs, outputs):
		self.inputs = inputs
		self.outputs = outputs


	def __call__(self, data):
		self.send(data)
		return self.recv()


	def send(self, data):
		self.outputs.send(data)


	def recv(self):
		return self.inputs.recv()


	@staticmethod
	def create():
		alpha_out, beta_in = Pipe()
		beta_out, alpha_in = Pipe()
		alpha_link = Link(alpha_in, alpha_out)
		beta_link = Link(beta_in, beta_out)
		return alpha_link, beta_link






if __name__ == '__main__':
	from time import sleep

	epochs = 3
	
	def network(link):
		data = link.recv()
		data = [3 + item for item in data]
		for e in range(epochs - 1):
			print(f'network epoch {e}\n')
			sleep(0.7)
			data = link(data)
			data = [3 + item for item in data]
			print('network send data', data)
		link.send(data)
		print('NETWORK MÜDE')
		#link.close()

	print(network)


	def mcts(link):
		data = 1
		for e in range(epochs):
			#print('worker started epoch', e)
			sleep(0.2)
			#print('worker finished')
			data = link(data)
			print('worker recv', data-1)
		print('WORKER MÜDE')


	with Interface(network, mcts) as inter:
		for worker in inter(3):
			worker.start()

