import numpy as np
from goban import Goban
import random



class Go:
	"""Implements a multiplayer MDP for Go"""
	def __init__(self, grid=(9, 9), komi=5.5):
		if isinstance(grid, tuple):
			board = np.zeros(grid)
		else:
			board = grid
		self.goban = Goban(board=board, komi=komi)
		self.state_class = Goban
		self.action_space = [(i, j) for i in range(self.goban.i) \
							for j in range(self.goban.j)] + [None]


	def __call__(self, action):
		"""plants the action and returns state and reward information"""
		result = self.goban(action)
		if result is False:
			raise Exception(f"The Action {action} is illegal!")
		self.goban = result
		terminal = self.goban.terminated
		return self.goban.x, terminal


	def __getattr__(self, attr):
		"""returns the ML compatible state representation"""
		if attr.lower() == 'x':
			return self.goban.x
		elif attr.lower() == 'state' or attr.lower() == 's':
			return self.goban
		else:
			raise AttributeError(f"The attribute '{attr}' is not defined.")


	def evaluate(self):
		return self.goban.tromp_taylor()


	def actions(self):
		"""returns a list of legal actions"""
		return self.goban.actions()



if __name__ == '__main__':
	game = Go()
	done, state = False, game.s
	while not done:
		actions = game.actions()
		action = random.choice(actions)
		state, done = game(action)
	print(game.trajectory())
	print(game.goban)
