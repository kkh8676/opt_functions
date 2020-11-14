import numpy as np

def made_v2(x1):
	minimum = -0.1
	maximum = 0.1

	def delta_func(x):
		return np.greater(x,minimum) * np.less(x,maximum) * (-3)

	return np.sin(x1) + delta_func(x1)

def main(job_id, params):
	return made_v2(params['x1'])
	