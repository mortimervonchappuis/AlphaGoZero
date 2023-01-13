import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, ReLU, Flatten, Dense, Layer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import numpy as np



class ResidualBlock(Layer):
	def __init__(self, n_kernels, CONFIG,**kwargs):
		super().__init__(**kwargs)
		self.activation = ReLU()
		self.conv_1 = SeparableConv2D(n_kernels, (3, 3), **CONFIG['conv'])
		self.batch_norm_1 = BatchNormalization()
		self.conv_2 = SeparableConv2D(n_kernels, (3, 3), **CONFIG['conv'])
		self.batch_norm_2 = BatchNormalization()


	def call(self, inputs):
		x = self.conv_1(inputs)
		x = self.batch_norm_1(x)
		x = self.activation(x)
		x = self.conv_2(x)
		x = self.batch_norm_2(x)
		x = self.activation(x)
		x = self.activation(x + inputs)
		return x


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}



class PolicyHead(Layer):
	def __init__(self, n_kernels, CONFIG, output_shape=None, **kwargs):
		super().__init__(**kwargs)
		self.n_kernels = n_kernels
		self.CONFIG = CONFIG
		self.activation = ReLU()
		self.conv = SeparableConv2D(n_kernels, (1, 1), **CONFIG['conv'])
		self.batch_norm = BatchNormalization()
		self.flatten = Flatten()
		if output_shape is not None:
			i, j, k = output_shape
			self.build((None, i, j, k))


	def build(self, input_shape):
		_, i, j, _ = input_shape
		self.dense = Dense(i * j + 1, activation='softmax', **self.CONFIG['dense'])
		super().build(input_shape)


	def call(self, inputs):
		p = self.conv(inputs)
		p = self.batch_norm(p)
		p = self.activation(p)
		p = self.flatten(p)
		p = self.dense(p)
		return p


	def get_config(self):
		base_config = super().get_config()
		return {**base_config, 
				'n_kernels': self.n_kernels, 
				'CONFIIG': self.CONFIG}



class ValueHead(Layer):
	def __init__(self, n_kernels, output_dim, CONFIG, **kwargs):
		print(kwargs)
		super().__init__(**kwargs)
		self.activation = ReLU()
		self.conv = SeparableConv2D(n_kernels, (1, 1), **CONFIG['conv'])
		self.batch_norm = BatchNormalization()
		self.flatten = Flatten()
		self.dense_1 = Dense(output_dim, activation='relu', **CONFIG['dense'])
		self.dense_2 = Dense(1, activation='tanh', **CONFIG['dense'])


	def call(self, inputs):
		v = self.conv(inputs)
		v = self.batch_norm(v)
		v = self.activation(v)
		v = self.flatten(v)
		v = self.dense_1(v)
		v = self.dense_2(v)
		return v


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}



class ResNet(Model):
	CONFIG = {'conv': {'kernel_regularizer': l2(1e-4), 'padding': 'same'}, 
			  'dense': {'kernel_regularizer': l2(1e-4)}}
	def __init__(self, 
				 n_kernels=256, 
				 n_residual_blocks=16, 
				 n_value_units=64, 
				 learning_rate=0.001,
				 **kwargs):
		super().__init__(**kwargs)
		self.n_kernels = n_kernels
		self.n_value_units = n_value_units
		self.conv = Conv2D(n_kernels, (3, 3), **self.CONFIG['conv'])
		self.batch_norm = BatchNormalization()
		self.activation = ReLU()
		self.residual_blocks = [ResidualBlock(n_kernels, self.CONFIG) \
								for n in range(n_residual_blocks - 1)]
		self.policy_head = PolicyHead(n_kernels, self.CONFIG)
		self.value_head = ValueHead(n_kernels, n_value_units, self.CONFIG)
		self.optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
		self.compile(optimizer=self.optimizer, 
					 loss=['categorical_crossentropy', 'mse'], 
					 loss_weights=[0.5, 0.5])


	def call(self, inputs):
		x = self.conv(inputs)
		x = self.batch_norm(x)
		x = self.activation(x)
		for residual_block in self.residual_blocks:
			x = residual_block(x)
		p = self.policy_head(x)
		v = self.value_head(x)
		return [p, v]


	def get_config(self):
		base_config = super().get_config()
		return {**base_config}


	def build(self, input_shape):
		self.policy_head.build(input_shape)
		self.value_head.build(input_shape)
		super().build(input_shape)


	def extend(self, n_value_units, output_shape, n_additional_blocks=1):
		self.n_value_units = n_value_units
		for n in range(n_additional_blocks):
			self.residual_blocks.append(ResidualBlock(self.n_kernels, self.CONFIG))
		self.policy_head = PolicyHead(self.n_kernels, self.CONFIG, output_shape)
		self.value_head = ValueHead(self.n_kernels, n_value_units, self.CONFIG)
		self.conv.trainable = False
		for residual_block in self.residual_blocks[:-n_additional_blocks]:
			residual_block.trainable = False


	def lock(self, n):
		if n:
			self.conv.trainable = False
		else:
			self.conv.trainable = True
		for residual_block in self.residual_blocks[:n-1]:
			residual_block.trainable = False
		for residual_block in self.residual_blocks[n-1:]:
			residual_block.trainable = True


	def reduce(self):
		self.policy_head = PolicyHead(2, self.CONFIG)
		self.value_head = ValueHead(2, self.n_value_units, self.CONFIG)







if __name__ == '__main__':
	shape = (1, 9, 9, 3)
	M = ResNet()
	#P, V = M.predict(np.ones(shape))
	#M.summary()
	#M.extend(1)
	#M.summary()
	#print(P, V)
	#M.save('test_model.pd')
	#M2 = ResNet.extend_model('test_model.pd', 2)
	#P, V = M2.predict(np.ones(shape))
	#M2.summary()
	from goban import Goban
	from utilities import factory


	Model = factory(ResNet, 
					n_kernels=256, 
					n_residual_blocks=6, 
					n_value_units=64)
	model = Model()
	#model.load_weights('trained_model.pd')
	#model.load_weights('RB_filler_model.pd')

	array0 = [[1, 1, 1,-1,-1,-1], 
			  [1, 1, 1,-1,-1,-1],
			  [1, 0, 1,-1,-1, 0], 
			  [1, 0, 1,-1,-1, 0], 
			  [1, 1, 1,-1,-1,-1], 
			  [1, 1, 1,-1,-1,-1]]

	array1 = [[1, 1, 1,-1,-1,-1], 
			  [1, 1, 1,-1,-1,-1],
			  [1, 1, 1,-1,-1,-1], 
			  [1, 0, 1,-1,-1, 0], 
			  [1, 1, 1,-1,-1,-1], 
			  [1, 1, 1,-1,-1,-1]]

	array2 = [[1, 1, 1,-1,-1,-1], 
			  [1, 1, 1,-1,-1,-1],
			  [1, 0, 1,-1,-1, 1], 
			  [1, 0, 1,-1,-1, 0], 
			  [1, 1, 1,-1,-1,-1], 
			  [1, 1, 1,-1,-1,-1]]

	array3 = [[0, 0, 0,-1,-1,-1], 
			  [0, 0, 0,-1,-1,-1],
			  [0, 0, 0,-1,-1,-1], 
			  [0, 0, 0,-1,-1,-1], 
			  [0, 0, 0,-1,-1,-1], 
			  [0, 0, 0,-1,-1,-1]]

	array4 = [[0, 0, 0, 0, 0, 0], 
			  [0, 0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0, 0], 
			  [0, 0, 0, 0, 0, 0], 
			  [0, 0, 0, 0, 0, 0], 
			  [0, 0, 0, 0, 0, 0]]

	array5 = [[0, 0, 0, 0, 0, 0], 
			  [0, 0, 0, 0, 0, 0],
			  [0, 0, 1, 0, 0, 0], 
			  [0, 0, 0, 0, 0, 0], 
			  [0, 0, 0, 0, 0, 0], 
			  [0, 0, 0, 0, 0, 0]]

	arrays = [array0, array1, array2, array3, array4, array5]


	def heatmap(array):
		i, j = array.shape
		min_v = np.min(array != 0)
		max_v = np.max(array)
		shades = ('  ', '░░', '▒▒', '▓▓', '██')
		vals = sorted(list(array.reshape(i * j)))
		vals = vals[::len(vals)//len(shades)]
		d = list(zip(vals[:-1], vals[1:], shades))
	
		def get(v):
			for a, b, c in d:
				if a <= v < b:
					return c
			return c
	
		top = ' ┌' + '──' * i + '┐\n │'
		mid = '│\n │'.join(''.join(get(v) for v in row) for row in array)
		bot = '│\n └' + '──' * i + '┘'
		return ''.join([top, mid, bot])
	
	def display(array, inv=False, black=True):
		g = Goban(np.array(array).T)
		g.colour = (1 if black else -1)
		if inv:
			g.board *= -1
		x = g.X.reshape((1, 6, 6, 3))
		#x = g.X.reshape((1, 7, 7, 3))

		P, ((V,),) = model.predict(x)
		print('GOBAN')
		print(g)
		print('TURN', 'WHITE' if g.colour == -1 else 'BLACK')
		print('VALUE')
		print(V)
		print('POLICY')
		#print(P[0,:-1].reshape(6, 6).T)
		print(heatmap(P[0,:-1].reshape(6, 6).T))
		#print(heatmap(P[0,:-1].reshape(7, 7).T))
		print(P[0,-1])

	#for array in arrays:
	#	display(array, inv=False, black=True)
	#	display(array, inv=True,  black=True)
	#	display(array, inv=False, black=False)
	#	display(array, inv=True,  black=False)

	array = [[0, 0, 0, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0]]
	for i in range(1, 13):
		model.load_weights(f'6x6/{i}/trained_model.pd')
		display(array4)

