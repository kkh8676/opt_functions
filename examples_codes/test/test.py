import numpy as np
import math

def test(x1):
	def delta_func(x1):
		return np.greater(x1,0.1) * np.less(x,0.1)*(-3)

	return np.sin(x1) + delta_func(x1)

def main(job_id, params):
	return test(params['x1'])
