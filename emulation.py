from agent import Agent
from threading import Thread
from utilities import TimeLogger, heatmap
from time import time



class Emulation:
	"""Server for MDP emulation"""
	TIMER = TimeLogger()
	def __init__(self, 
				 MDP_class, 
				 n_tree_nodes, 
				 n_sampling_moves, 
				 n_threads, 
				 n_moves, 
				 replay_buffer=None, 
				 temperatur=1, 
				 **kwargs):
		self.replay_buffer = replay_buffer
		self.MDP_class = MDP_class
		self.n_tree_nodes = n_tree_nodes
		self.n_sampling_moves = n_sampling_moves
		self.n_threads = n_threads
		self.n_moves = n_moves
		self.temperatur = temperatur
		self.Z_AVRG = 0
		self.Z = 0
		self.N = 0
		self.eta = 0.03


	@TIMER.log
	def play(self, link, seed):
		trajectory = []
		MDP = self.MDP_class()
		agent = Agent(MDP_class=self.MDP_class, 
					  state=MDP.S, 
					  n_sammpling_moves=self.n_sampling_moves, 
					  link=link, 
					  n_threads=self.n_threads, 
					  temperatur=self.temperatur, 
					  seed=seed + int(time()))
		Vs, X = [], MDP.S.X
		for t in range(self.n_moves):
			PI, A = agent(self.n_tree_nodes)
			Vs.append(agent.mcts.root.V)
			trajectory.append((X, PI))
			X, terminal = MDP(A)
			if terminal: break
		Z = MDP.evaluate()
		trajectory = [(X, PI, Z*((-1)**i)) for i, (X, PI) in enumerate(trajectory[:-1])]
		X, PI, Z_end = trajectory[-1]
		V_end = Vs[-2]
		self.Z_AVRG = (1 - self.eta) * self.Z_AVRG + self.eta * Z
		print(f"""{MDP.S}
BLACK VICTORIES: {round(self.Z_AVRG, 3)}
LEN OF THE GAME: {len(trajectory)}
{'BLACKS' if Z == 1 else 'WHITES'} VICTORY
EVALUATION ERROR: {round(Z_end + V_end, 2)}""")
		self.replay_buffer.extend(trajectory[:-3])


	def eval(self, link, PID, score, phase='alpha'):
		flag = True
		while True:
			# START GAME
			MDP = self.MDP_class()
			alpha_agent = Agent(MDP_class=self.MDP_class, 
								state=MDP.S, 
								n_sammpling_moves=self.n_sampling_moves, 
								link=link, 
								n_threads=self.n_threads, 
								temperatur=self.temperatur,
								seed=(int(time()) + PID),
								verbose=(PID == 0))
			
			beta_agent = Agent(MDP_class=self.MDP_class, 
							   state=MDP.S, 
							   n_sammpling_moves=self.n_sampling_moves, 
							   link=link, 
							   n_threads=self.n_threads, 
							   temperatur=self.temperatur, 
							   seed=(int(time()) - PID),
							   randomize=False,
							   verbose=(PID == 0))
			
			alpha_pass = 0
			beta_pass  = 0
			while True:
				PI, A = alpha_agent(self.n_tree_nodes)
				beta_agent.mcts.forward(A)
				X, terminal = MDP(A)
				if A is None:
					alpha_pass += 1
				if terminal: break

				PI, A = beta_agent(self.n_tree_nodes)
				alpha_agent.mcts.forward(A)
				X, terminal = MDP(A)
				if A is None:
					beta_pass += 1
				if terminal: break

			# CREATE TRAJECTORY
			print(MDP.S, alpha_pass, beta_pass)
			Z = MDP.evaluate()
			if flag:
				score.append(Z)
				flag = False
			


	def run(self, *args, **kwargs):
		args = args
		kwargs = kwargs
		while True:
			self.play(*args, **kwargs)

