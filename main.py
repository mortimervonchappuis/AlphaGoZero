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
from simulation import GoSimulation



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
	# SWITCH BETWEEN THE TWO LINES TO TOGGLE ARTIFICIAL OPPONENT
	Emulation.play(agent=agent, turn=False)
	#Emulation.play(turn=False)