from time import time
import numpy as np



def factory(obj_class, func=None, **kwargs):
	if func:
		return lambda kwargs=kwargs: func(obj_class(**kwargs))
	else:
		return lambda kwargs=kwargs: obj_class(**kwargs)



def group(n, *args):
	return tuple([arg[i*len(arg)//n:(i+1)*len(arg)//n] for i in range(n)] for arg in args)



def softmax(x):
	z = np.exp(x)
	return z/np.sum(z)


def temperatur(x, t):
	z = x**(1/t)
	return z/np.sum(z)



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
			if a <= v < b or a == v:
				return c
		return c

	top = ' ┌' + '──' * i + '┐\n │'
	mid = '│\n │'.join(''.join(get(v) for v in row) for row in array)
	bot = '│\n └' + '──' * i + '┘'
	return ''.join([top, mid, bot])


class TimeLogger:
	def __init__(self, eta=0.3):
		self.eta = eta
		self.book = {}


	def __repr__(self):
		OFFSET = 8
		top = 'FUNCTION\tAVERAGE RUNTIME'
		mid = '\n'.join(func + (OFFSET - max(0, len(func))) * ' ' + \
						f'\t{dt:.12f} sec' for func, dt in self.book.items())
		return f'{top}\n{mid}'


	def __setitem__(self, func, dt_new):
		if func not in self.book:
			self.book[func] = dt_new
		else:
			dt_total = self.book[func]
			self.book[func] = (1 - self.eta) * dt_total + self.eta * dt_new


	def __getitem__(self, attr):
		if attr in self.book:
			return self.book[attr]
		elif '__name__' in attr.__dir__():
			return self.book[attr.__name__]
		raise AttributeError(f'Function {attr} not logged')


	def log(self, f_raw):
		def f_mod(*args, logger=self):
			t0 = time()
			results = f_raw(*args)
			t1 = time()
			dt = t1 - t0
			self[f_raw.__name__] = dt
			return results
		f_mod.__name__ = f_raw.__name__
		return f_mod



if __name__ == '__main__':
	timer = TimeLogger()


	@timer.log
	def fib_rec(n):
		if n <= 1:
			return 1 
		else:
			return fib_rec(n-1) + fib_rec(n-2)


	@timer.log
	def fib_iter(n):
		a, b = 0, 1
		for i in range(n):
			a, b = b, a + b
		return b

	from math import log

	@timer.log
	def ld(x):
		return log(x)


	for i in range(20):
		print(fib_iter(i), fib_rec(i))
		print(ld(i + 1))
	print()
	print(timer)



	l = list(range(12))
	print(group(3, l, l))

