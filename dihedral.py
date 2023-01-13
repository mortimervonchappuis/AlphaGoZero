import numpy as np
import random
from time import time



class DihedralReflector:
	def __init__(self, seed=None):
		random.seed(seed or int(time()*100))
		self.built = False


	def __call__(self, X, PI=None):
		# BUILD
		if not self.built:
			self.i, self.j = X.shape[:2]
			self.built = True
		omega = random.randrange(2)
		phi = random.randrange(4)
		R = (omega, phi)

		# REFLECTION
		if omega:
			X = X[::-1,...]

		# ROTATION
		X = np.rot90(X, k=phi)
		
		# OPTIONAL
		if PI is not None:
			passing = PI[-1]
			PI = np.array(PI[:-1]).reshape(self.i, self.j)
	
			# ROTATION
			PI = np.rot90(PI, phi)
	
			# REFLECTION
			if omega:
				PI = PI[::-1,:]
			PI = list(PI.reshape(self.i * self.j)) + [passing]
			return R, X, PI
		return R, X


	def inv(self, P, R, X=None):
		# INIT
		omega, phi = R
		passing = P[-1]
		#print('PRE', P)
		P = np.array(P[:-1]).reshape(self.i, self.j)
		#print('POST', P)

		# ROTATION
		P = np.rot90(P, 4 - phi)
		if X is not None:
			X = np.rot90(X, 4 - phi)

		# REFLECTION
		if omega:
			P = P[::-1,:]
			if X is not None:
				X = X[::-1,:,:]

		P = list(P.reshape(self.i * self.j)) + [passing]
		if X is not None:
			return P, X
		else:
			return P




if __name__ == '__main__':
	from goban import Goban

	g = Goban(np.zeros((5, 5)))
	g = g((1, 1))
	g = g((3, 3))
	g = g((1, 3))

	reflector = DihedralReflector(0)
	print(g)
	print(-g.X[:,:,0].T + g.X[:,:,1].T)
	print(g.X.shape)
	reflection, board = reflector(g.X)
	A = -board[:,:,0] + board[:,:,1]
	print(A.T)
	print(reflection)
	array = list(A.reshape(25)) + [2]
	print('PRE PRE', array)
	reconstruction = reflector.inv(array, reflection)
	print('POST POST', reconstruction)
	print(np.array(reconstruction[:-1]).reshape(5, 5).T)
	print(reconstruction[-1])