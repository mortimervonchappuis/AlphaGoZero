import numpy as np
from go import Go
from goban import Goban
from tkinter import *
from PIL import Image, ImageTk
from playsound import playsound
from threading import Thread
from time import time
from model import Model
from utilities import factory
from agent import AutonomousAgent
from model import ResNet



class GoSimulation:
	'Implements a small GUI to play Go'
	def __init__(self, size=(19, 19), komi=5.5, handicap=0):
		self.size = self.i, self.j = size
		self.komi = komi
		self.handicap = handicap
		self.goban = Goban(board=self._board(), komi=komi, colour=1)


	def __call__(self, pos):
		if pos is not None:
			i, j = pos
			if i not in range(self.i) or j not in range(self.j):
				return False, None
		legal, afterstate = self.goban.move(pos)
		if legal:
			self.goban = afterstate
			if pos is not None:
				Thread(target=lambda: playsound('resources/sounds/placement.mp3')).start()
				self.display()
				return True
		return False


	def _board(self):
		offset = 2 if self.i <= 11 or self.j <= 11 else 3
		board = np.zeros(self.size)
		positions = {'NW': (offset, offset), 
					 'N':  (offset, self.j//2), 
					 'NE': (offset, self.j - offset - 1), 
					 'W':  (self.i//2, offset), 
					 'C':  (self.i//2, self.j//2), 
					 'E':  (self.i//2, self.j - offset - 1), 
					 'SW': (self.i - offset - 1, offset), 
					 'S':  (self.i - offset - 1, self.j//2), 
					 'SE': (self.i - offset - 1, self.j - offset - 1)}
		handicaps = {0: (), 
					 1: ('NE',), 
					 2: ('NE', 'SW'), 
					 3: ('NE', 'SW', 'SE'), 
					 4: ('NE', 'SW', 'SE', 'NW'), 
					 5: ('NE', 'SW', 'SE', 'NW', 'C'), 
					 6: ('NE', 'SW', 'SE', 'NW', 'W', 'E'), 
					 7: ('NE', 'SW', 'SE', 'NW', 'W', 'E', 'C'), 
					 8: ('NE', 'SW', 'SE', 'NW', 'W', 'E', 'N', 'S'), 
					 9: ('NE', 'SW', 'SE', 'NW', 'W', 'E', 'N', 'S', 'C'),}
		for point in handicaps[self.handicap]:
			board[positions[point]] = 1
		return board


	def img(self):
		img = self.board_img.copy()
		for x, y in self.goban._indicies(self.goban.black):
			img.paste(self.black_img, (x * 32 + 16, y * 32 + 16), self.black_img)
		for x, y in self.goban._indicies(self.goban.white):
			img.paste(self.white_img, (x * 32 + 16, y * 32 + 16), self.white_img)
		return img


	def create_grid(self):
		board_img = Image.open('resources/imgages/board_background.png')
		board_img = board_img.crop((0, 0, 32*(self.j + 1), 32*(self.i + 1)))
		background = np.array(board_img.getdata()).astype(np.float64)
		background = background.reshape(board_img.size[0], board_img.size[1], 4)
		darkening = 0.7
		for i in range(32, (self.i+1)*32, 32):
			background[32:self.j*32,i-2:i+2,:2] *= darkening
			background[31:self.j*32+1,i-1:i+1,:2] = 0
		for j in range(32, (self.j+1)*32, 32):
			darkening = 0.8
			background[j-2:j+2,32:self.i*32,:2] *= darkening
			background[j-1:j+1,31:self.i*32+1,:2] = 0
		background[30:32,30:32,:2] *= darkening
		background[self.j*32:self.j*32+2,30:32,:2] *= darkening
		background[30:32,self.i*32:self.i*32+2,:2] *= darkening
		background[self.j*32:self.j*32+2,self.i*32:self.i*32+2,:2] *= darkening
		return Image.fromarray(background.astype(np.uint8))


	def mouse_click(self, event):
		x, y = event.x, event.y
		pos = (x - 16)//32, (y - 16)//32
		if self(pos):
			print('TROMP TAYLOR', self.goban.tromp_taylor())
			print(self.goban)
			if self.agent is not None:
				print('TROMP TAYLOR', self.goban.tromp_taylor())
				self.play_oponent_move(pos)


	def play_oponent_move(self, pos=None):
		if pos is not None:
			self.agent.mcts.forward(pos)
		t0 = time()
		PI, A = self.agent()
		t1 = time()
		print('TIME', round(t1 - t0, 1), 'sec')
		print(self.agent.mcts.root)
		self(A)
		Thread(target=lambda: playsound('resources/sounds/placement.mp3')).start()
		self.display()
 

	def back_space(self, event):
		predecessor = self.goban.parent
		if predecessor is not None:
			self.goban = predecessor
			self.display()


	def space(self, event):
		if self(None):
			self.display()


	def display(self):
		img = ImageTk.PhotoImage(self.img())
		self.tk_board.configure(image=img)
		self.tk_board.image = img
		self.master.update()


	def play(self, agent=None, turn=False):
		self.agent = agent
		'A minimalist GUI'
		self.board_img = self.create_grid()
		self.black_img = Image.open('resources/imgages/black-b2.png')
		self.white_img = Image.open('resources/imgages/white-b3.png')

		self.master = Tk()
		self.master.title('Shodan Go-Engine')
		tk_board_img = ImageTk.PhotoImage(self.board_img)
		self.tk_board = Label(self.master, image=tk_board_img)
		self.tk_board.pack()
		self.tk_board.bind('<ButtonPress-1>', func=self.mouse_click)
		self.master.bind('<BackSpace>', func=self.back_space)
		self.master.bind('<space>', func=self.space)
		self.master.bind('<Escape>', func=quit)
		self.display()
		if turn:
			self.play_oponent_move()
		self.master.mainloop()



if __name__ == '__main__':
	MDP_class = factory(Go, grid=(9, 9), komi=5.5)
	MDP = MDP_class()

	def load(model):
		#model.reduce()
		model.load_weights('final.pd')
		return model


	model = factory(ResNet,
					n_kernels=256, 
					func=load, 
					n_residual_blocks=9, 
					learning_rate=0.0001,
					n_value_units=128)

	agent = AutonomousAgent(MDP_class=MDP_class,
							state=MDP.S,
							n_sammpling_moves=0,
							model=model,
							C=5,
							n_threads=16,
							n_tree_nodes=800)

	Emulation = GoSimulation(size=(9, 9), handicap=0)
	Emulation.play(agent=agent, turn=False)
	#Emulation.play(turn=False)