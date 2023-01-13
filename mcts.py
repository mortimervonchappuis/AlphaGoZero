import math
import numpy as np
from threading import Thread
from multiprocessing import Process, Pipe
from time import time
from dihedral import DihedralReflector



class MCTS:
	def __init__(self, MDP, root, action_space, link, n_threads, C=5):
		self.MDP = MDP
		self.root = root
		self.root.tree = self
		self.action_space = action_space
		self.link = link
		self.n_threads = n_threads
		self.C = C
		self.dirichlet_alpha = 0.03# 0.03
		self.dirichlet_epsilon = 0.25
		self.virtual_loss = 2 ### 3
		self.reflector = DihedralReflector()


	def __call__(self, n_tree_nodes):
		inter_links, thread_workers = [], []

		# EXPLORATION NOISE
		if self.root.edges:
			actions = self.root.actions()
			noise = np.random.gamma(self.dirichlet_alpha, 1, len(actions))
			epsilon = self.dirichlet_epsilon
			for n, A in zip(noise, actions):
				self.root[A].P = self.root[A].P * (1 - epsilon) + n * epsilon

		# INITIALIZE WORKERS
		for n in range(n_tree_nodes//self.n_threads):
			self.search()

		# CREATE PI
		PI = np.zeros((len(self.action_space)))
		for child in self.root.children():
			PI[self[child.A]] = child.N
		return PI


	def __getitem__(self, key):
		return self.action_space.index(key)


	def search(self):
		# SELECTION
		nodes = []
		for n in range(self.n_threads):
			node = self.root
			while node.edges is not None and not node.S.terminated:
				ucb = lambda n: n.Q + self.C * n.P * \
					  math.sqrt(n.parent.N) / (1 + n.N)
				# VIRTUAL LOSS
				node = max(node.children(), key=ucb)
				node.N += self.virtual_loss
				node.W -= self.virtual_loss
			nodes.append(node)

		# AUGMENT SYMMETRIES
		Xs, Rs = [], []
		for node in nodes:
			R, X = self.reflector(node.S.X)
			Xs.append(X)
			Rs.append(R)

		# EVALUATION
		Ps, Vs = self.link(Xs)
		for node, R, P, (V,) in zip(nodes, Rs, Ps, Vs):
			# REVERSE SYMMETRIES
			P = self.reflector.inv(P, R)

			# EXSPANSION
			if not node.S.terminated:
				node.expand(P, V)

			# BACKUP
			while node.parent is not None:
				V = -V
				node.N += 1 - self.virtual_loss
				node.W += V + self.virtual_loss
				#node.Q = node.W / node.N
				node = node.parent


	def forward(self, A):
		if self.root.edges:
			self.root = self.root[A]
		else:
			self.root = Node(state=self.root.S(A), 
							 parent=self, 
							 tree=self, 
							 action=A)
		self.root.parent = None



class MCTSasync:
	def __init__(self, MDP, root, action_space, model, n_threads, C=5):
		self.MDP = MDP
		self.root = root
		self.root.tree = self
		self.action_space = action_space
		self.model = model
		self.n_threads = n_threads
		self.C = C
		self.dirichlet_alpha = 0.03# 0.03
		self.dirichlet_epsilon = 0.25
		self.virtual_loss = 2 ### 3
		self.reflector = DihedralReflector()
		self.reflector(self.MDP().S.X)


	def __call__(self, n_tree_nodes):
		# EXPLORATION NOISE
		if self.root.edges:
			actions = self.root.actions()
			noise = np.random.gamma(self.dirichlet_alpha, 1, len(actions))
			epsilon = self.dirichlet_epsilon
			for n, A in zip(noise, actions):
				self.root[A].P = self.root[A].P * (1 - epsilon) + n * epsilon


		# INITIALIZE WORKERS
		inflow_active   = [Pipe() for _ in range(self.n_threads)]
		inflow_passive  = [Pipe() for _ in range(self.n_threads)]
		outflow_active  = [Pipe() for _ in range(self.n_threads)]
		outflow_passive = [Pipe() for _ in range(self.n_threads)]
		
		inflow_active_input,   inflow_active_output   = zip(*inflow_active)
		inflow_passive_input,  inflow_passive_output  = zip(*inflow_passive)
		outflow_active_input,  outflow_active_output  = zip(*outflow_active)
		outflow_passive_input, outflow_passive_output = zip(*outflow_passive)

		# WORKER FUNCTION
		def w_func(reflector, state_class, inflow, outflow):
			for n in range(math.ceil(n_tree_nodes/(self.n_threads*2))):
				P, V, X, R = inflow.recv()
				P, X = reflector.inv(P, R, X)
				S = state_class(X)
				expansion = list(S)
				outflow.send([expansion, P, V])

		active_workers  = [Process(target=w_func, args=(self.reflector, 
														self.MDP().state_class, 
														inflow, 
														outflow)) 
			for inflow, outflow in zip(inflow_active_output, outflow_active_input)]
		passive_workers = [Process(target=w_func, args=(self.reflector, 
														self.MDP().state_class, 
														inflow, 
														outflow)) 
			for inflow, outflow in zip(inflow_passive_output, outflow_passive_input)]
		
		# INITIALIZE NETWORK
		network_input, network_output = Pipe()

		# NETWORK FUNCTION
		def n_func(pipe, active, passive, model):
			network = model()
			for n in range(n_tree_nodes//self.n_threads):
				Xs, Rs = pipe.recv()
				X = np.array(Xs)
				Ps, Vs = network.predict(X)
				for worker, P, V, X, R in zip(active, Ps, Vs, Xs, Rs):
					worker.send([P, float(V), X, R])
				active, passive = passive, active

		network_server = Process(target=n_func, args=(network_output, 
													  inflow_active_input, 
													  inflow_passive_input, 
													  self.model))

		# START PROCESSES
		network_server.start()
		for worker in active_workers:
			worker.start()
		for worker in passive_workers:
			worker.start()

		# START SEARCH
		active = True
		for n in range(n_tree_nodes//self.n_threads):
			outflow = outflow_active_output if active else outflow_passive_output
			self.search(outflow, network_input)
			active = not active

		# CREATE PI
		PI = np.zeros((len(self.action_space)))
		for child in self.root.children():
			PI[self[child.A]] = child.N
		return PI


	def __getitem__(self, key):
		return self.action_space.index(key)


	def search(self, active, network):
		# SELECTION
		nodes = []
		for n in range(self.n_threads):
			node = self.root
			while node.edges is not None and not node.S.terminated:
				ucb = lambda n: n.Q + self.C * n.P * \
					  math.sqrt(n.parent.N) / (1 + n.N)
				# VIRTUAL LOSS
				node = max(node.children(), key=ucb)
				node.N += self.virtual_loss
				node.W -= self.virtual_loss
			nodes.append(node)

		# AUGMENT SYMMETRIES
		Xs, Rs = [], []
		for node in nodes:
			R, X = self.reflector(node.S.X)
			Xs.append(X)
			Rs.append(R)

		# EVALUATION
		network.send([Xs, Rs])
		expansions, Ps, Vs = [], [], []
		for pipe in active:
			data = pipe.recv()
			expansion, P, V = data
			expansions.append(expansion)
			Ps.append(P)
			Vs.append(V)
		for node, expansion, P, V in zip(nodes, expansions, Ps, Vs):
			# EXSPANSION
			if not node.S.terminated:
				node.update(expansion, P, V)

			# BACKUP
			while node.parent is not None:
				V = -V
				node.N += 1 - self.virtual_loss
				node.W += V + self.virtual_loss
				node = node.parent



	def forward(self, A):
		if self.root.edges:
			self.root = self.root[A]
		else:
			self.root = Node(state=self.root.S(A), 
							 parent=self, 
							 tree=self, 
							 action=A)
		self.root.parent = None



class Node:
	def __init__(self, state, tree=None, parent=None, action=None, prior=1.0):
		self.edges = None
		self.tree = tree
		self.parent = parent
		self.A = action
		self.S = state
		self.P = prior
		self.V = 0
		self.W = 0
		self.N = 0


	def __getattr__(self, attr):
		if attr == 'Q':
			if self.N:
				return self.W/self.N
			else:
				return 0


	def __eq__(self, other):
		return self.S == other.S


	def children(self):
		return self.edges.values()


	def actions(self):
		return self.edges.keys()


	def __getitem__(self, key):
		if isinstance(key, Node):
			for action, node in self.edges.iteritems():
				if Node is key:
					return action
		else:
			return self.edges[key]


	def expand(self, Ps, V):
		self.V = float(V)
		self.Ps = Ps
		self.edges = {A: Node(state=S, 
							  tree=self.tree, 
							  parent=self, 
							  action=A, 
							  prior=Ps[self.tree[A]]) \
						for A, S in self.S}


	def __repr__(self):
		padding = lambda x: str(x) + '\t' if len(str(x)) < 3 else str(x) 
		goban = str(self.S)
		stats = f"P={round(self.P, 3)}\tQ={round(self.Q, 3)}\tV={round(self.V, 2)}\tW={round(self.W, 2)}\tN={self.N}\tA={self.A}"
		if self.edges:
			children = '\n'.join(f"P={padding(round(float(n.P), 3))}\tA={n.A}\tN={n.N}\tV={padding(round(n.V, 2))}\tQ={padding(round(n.Q, 2))}\tW={padding(round(n.W, 2))}" for n in self.children() if n.N)
		else:
			children = '{}'
		return f"""NODE
{goban}
STATS
{stats}\tC = {'BLACK' if self.S.blacks_turn() else 'WHITE'}
CHILDREN
{children}"""


	def update(self, expansion, Ps, V):
		self.V = float(V)
		self.Ps = Ps
		self.edges = {A: Node(state=S, 
							  tree=self.tree, 
							  parent=self, 
							  action=A, 
							  prior=float(Ps[self.tree[A]])) \
						for A, S in expansion}
		# CHECKING FOR KO
		ko_moves = set()
		for A, node in self.edges.items():
			if self.parent is None:
				break
			parent = self.parent
			for i in range(8):
				if parent == node:
					ko_moves.add(A)
		for A in ko_moves:
			del self.edges[A]
