import numpy as np
from utilities import TimeLogger



class Goban:
	"""a class that implements a game state"""
	def __init__(self, 
				 board=np.zeros((19, 19)), 
				 komi=5.5, 
				 colour=1, 
				 parent=None, 
				 action=None, 
				 action_space=None, 
				 terminated=False):
		if len(board.shape) == 2:
			self.i, self.j = self.size = board.shape
			self.board = board
			self.colour = colour
		else:
			self.i, self.j = self.size = board.shape[:2]
			# BLACK
			if np.all(board[:,:,2] == 1):
				self.board = board[:,:,0] - board[:,:,1]
				self.colour = 1
			# WHITE
			else:
				self.board = board[:,:,1] - board[:,:,0]
				self.colour = -1
		self.komi = komi
		self.parent = parent
		self.action = action
		self.terminated = terminated
		self.action_space = action_space or [(i, j) for i in range(self.i) \
											 for j in range(self.j)] + [None]
		self.killed = False


	def __call__(self, action):
		"""performs the action on a copy of the board and returns it"""
		if self.terminated:
			return False
		if action is not None and (action not in self.action_space or self[action] != 0):
			return False
		legal, afterstate = self.move(action)
		if action is None and self.action is None and self.parent is not None:
			afterstate.terminated = True
		if action is None or legal:
			return afterstate
		else:
			return False


	def __getitem__(self, key):
		"""gets item from boardstate"""
		return self.board[key]


	def __setitem__(self, key, value):
		"""sets values in the boardstate"""
		self.board[key] = value


	def __getattr__(self, attr):
		"""returns boards for black and white"""
		if attr.lower() == 'x':
			if self.blacks_turn():
				x = np.array([self.black, self.white, np.ones(self.size)])
			else:
				x = np.array([self.white, self.black, np.zeros(self.size)])
			return x.transpose((1, 2, 0))
		elif attr.lower() == 'black':
			black = self.board.copy()
			black[black == -1] = 0
			return black
		elif attr.lower() == 'white':
			white = self.board.copy()
			white[white == 1] = 0
			return -white
		else:
			raise AttributeError(f"The attribute '{attr}' is not defined.")


	def __iter__(self):
		if self.terminated:
			return
		for A in self.action_space:
			legal, S = self.move(A)
			if legal:
				yield A, S


	def __bool__(self):
		"""returns if the game is still over"""
		return not self.terminated


	def __eq__(self, other):
		return (self.board == other.board).all()


	def __str__(self):
		symb = {0: ' ', 1: '◻', -1: '◼'}
		top = '┌─' + self.j * '──' + '┐\n'
		mid = '\n'.join('│ ' + ' '.join(symb[s] for s in row) + ' │' for row in self.board.T)
		bottom = '\n└─' + self.j * '──' + '┘'
		return top + mid + bottom


	def __contains__(self, pos):
		"""returns wether the position is in the range of the boards grid"""
		i, j = pos
		return i in range(self.i) and j in range(self.j)


	def action_map(self):
		if self.terminated:
			return np.zeros(self.size).astype(bool)
		critical_map = (self._indiv_liberties() <= 2)
		#row, col = np.indices(self.size)
		critical_intersections = self._indicies(critical_map)
		for pos in critical_intersections:
			afterstate = self.placement(pos)
			if afterstate._legal():
				critical_map[pos] = False
		legal = self._empty()
		legal[critical_map] = False
		return legal


	def actions(self):
		if self.terminated:
			return []
		else:
			return self._indicies(self.action_map()) + [None]


	def move(self, pos):
		if pos is not None and self[pos] != 0:
			return False, None
		afterstate = self.copy()
		afterstate.parent = self
		afterstate.action = pos
		afterstate._toggle()
		if pos is not None:
			afterstate[pos] = self.colour
			for neighbor in afterstate._neighbors(pos):
				if afterstate[neighbor] == afterstate.colour:
					group = afterstate._group(neighbor)
					if not afterstate._alive(group):
						afterstate._remove(group)
						afterstate.killed = True
		else:
			if self.action is None and self.parent is not None:
				afterstate.terminated = True
			return True, afterstate 
		if not afterstate._legal():
			return False, None
		return True, afterstate


	def _legal(self):
		return not self._suicide(self.action) and not self._ko()


	def _alive(self, group):
		indiv_liberties = self._indiv_liberties()
		return 0 < sum(indiv_liberties[pos] for pos in group)


	def _suicide(self, pos):
		return not self.killed and not self._alive(self._group(pos))


	def _ko(self):
		predecessors = []
		predecessor = self.parent
		for _ in range(8):
			if predecessor is None:
				break
			else:
				predecessors.append(predecessor)
				predecessor = predecessor.parent
		return any(self == predecessor for predecessor in predecessors)


	def _group(self, pos):
		colour, frontier, group = self[pos], [pos], []
		while frontier:
			node = frontier.pop()
			if self[node] == colour and node not in group:
				group.append(node)
				for neighbor in self._neighbors(node):
					if neighbor not in frontier and neighbor not in group:
						frontier.append(neighbor)
		return group


	def _remove(self, group):
		for pos in group:
			self[pos] = 0


	def tromp_taylor(self):
		empty = self._empty()
		black_idx = self._indicies(self.black)
		white_idx = self._indicies(self.white)
		expanded, groups = [], []
		for intersection in self._indicies(empty):
			if intersection not in expanded:
				group = self._group(intersection)
				expanded.extend(group)
				groups.append(group)

		black_moku, white_moku = np.sum(self.black), self.komi + np.sum(self.white)
		for group in groups:
			neighbors = []
			for intersection in group:
				for neighbor in self._neighbors(intersection):
					if neighbor not in group:
						neighbors.append(self[neighbor])
			if all(stone == 1 for stone in neighbors):
				black_moku += len(group)
			elif all(stone == -1 for stone in neighbors):
				white_moku += len(group)
		return (1 if black_moku > white_moku else -1)


	def blacks_turn(self):
		return self.colour == 1

	
	def whites_turn(self):
		return self.colour == -1


	def _toggle(self):
		self.colour *= -1


	def copy(self):
		return Goban(
			board=self.board.copy(), 
			komi=self.komi, 
			colour=self.colour, 
			parent=self.parent,
			action=self.action,
			action_space=self.action_space, 
			terminated=self.terminated)


	def _indicies(self, idx_map):
		row, col = np.indices(self.size)
		return [(i, j) for i, j in zip(row[idx_map.astype(bool)], 
									   col[idx_map.astype(bool)])]


	def _empty(self):
		"""returns an ma-p of empty intersections"""
		return self.board == 0


	def _indiv_liberties(self):
		if not hasattr(self, 'indiv_liberties'):
			"""returns a map of the liberties of every individual intersection"""
			empty = self._empty()
			indiv_liberties = np.zeros(self.size)
			indiv_liberties[:self.i-1,:] += empty[1:,:]
			indiv_liberties[1:,:] += empty[:self.i-1,:]
			indiv_liberties[:,:self.j-1] += empty[:,1:]
			indiv_liberties[:,1:] += empty[:,:self.j-1]
			self.indiv_liberties = indiv_liberties
		return self.indiv_liberties
	

	def _neighbors(self, pos):
		for neighbor in {self.west(pos), 
						 self.east(pos), 
						 self.north(pos), 
						 self.south(pos)}:
			if neighbor in self:
				yield neighbor


	@staticmethod
	def west(pos):
		i, j = pos
		return i, j - 1


	@staticmethod
	def east(pos):
		i, j = pos
		return i, j + 1


	@staticmethod
	def north(pos):
		i, j = pos
		return i - 1, j


	@staticmethod
	def south(pos):
		i, j = pos
		return i + 1, j



if __name__ == '__main__':
	s = np.array(
		[
			[ 1, 0,-1, 1, 0], 
			[-1,-1,-1, 1, 0], 
			[-1, 0,-1, 1, 1], 
			[ 0,-1,-1, 1, 0],
		])
	g = Goban(np.zeros((5, 5)), komi=0.5)
	print(g.x.shape)
	
	def display(x):
		print('DISPLAY')
		print(abs(x[:,:,0]))
		print(abs(x[:,:,1]))
		print(abs(x[:,:,2]))
	display(g.X)
	g = g((1, 1))
	display(g.X)
	g = g((2, 2))
	display(g.X)
	#g = g((3, 3))
	display(g.X)
	print(g.tromp_taylor())
	print(g)
	g = g(None)
	print(g.tromp_taylor())
	print(g)
	#print(g._indicies(np.zeros((4, 5))))
