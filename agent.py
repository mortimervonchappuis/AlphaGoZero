from mcts import MCTS, MCTSasync, Node
from utilities import temperatur, heatmap
from multiprocessing import Manager, Process
from interface import Interface
#from networkserver import NetworkServer
#from emulation import Emulation
import numpy as np


class Agent:
	def __init__(self, 
				 MDP_class, 
				 state, 
				 n_sammpling_moves, 
				 link, 
				 n_threads, 
				 temperatur, 
				 seed=None,
				 randomize=False, 
				 verbose=False):
		if seed is not None:
			np.random.seed(seed)
		self.MDP = MDP_class
		self.temperatur = temperatur
		self.action_space = MDP_class().action_space
		self.mcts = MCTS(MDP=self.MDP, 
						 root=Node(state), 
						 action_space=self.action_space, 
						 link=link, 
						 n_threads=n_threads)
		self.n_sammpling_moves = n_sammpling_moves
		self.randomize = randomize
		self.verbose = verbose
		self.t = 0


	def __call__(self, n_tree_nodes):
		logits = self.mcts(n_tree_nodes)
		PI = temperatur(logits, t=self.temperatur)
		actions = np.array(self.action_space, dtype=object)
		if self.randomize:
			A = np.random.choice(list(self.mcts.root.actions()))
		else:
			if self.t < self.n_sammpling_moves:
				A = np.random.choice(actions, p=PI)
			else:
				A = self.action_space[np.argmax(PI)]
		if self.verbose:
			print()
			if self.mcts.root.S.blacks_turn():
				print('BLACKS TURN')
			else:
				print('WHITES TURN')
			print('Ps')
			Ps_array = np.array(self.mcts.root.Ps[:-1]).reshape((9, 9)).T
			print(heatmap(Ps_array))
			print('PI')
			PI_array = PI[:-1].reshape((9, 9)).T
			print(heatmap(PI_array))
			print(self.mcts.root)
		self.mcts.forward(A)
		self.t += 1
		return PI, A



class AutonomousAgent(Agent):
	def __init__(self,
				 MDP_class,
				 state,
				 n_sammpling_moves,
				 model,
				 n_threads,
				 n_tree_nodes,
				 temperatur=1,
				 C=5,
				 seed=None,
				 randomize=False,
				 verbose=False):
		self.MDP = MDP_class
		self.temperatur = temperatur
		self.action_space = MDP_class().action_space
		self.model = model
		self.mcts = MCTSasync(MDP=self.MDP, 
							   root=Node(state), 
							   action_space=self.action_space, 
							   model=self.model, 
							   C=C,
							   n_threads=n_threads)
		self.n_sammpling_moves = n_sammpling_moves
		self.n_tree_nodes = int(ceil(n_tree_nodes/n_threads)) * n_threads
		self.randomize = randomize
		self.verbose = verbose
		self.t = 0


	def __call__(self):
		return super().__call__(self.n_tree_nodes)
