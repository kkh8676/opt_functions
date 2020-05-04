import numpy as np
import math

def mix_cosines(x1,x2):
	u = 1.6 * x1 - 0.5
	v = 1.6 * x2 - 0.5

	element2 = u**2 + v**2 - 0.3 * np.cos(3*np.pi*u) - 0.3 * np.cos(3*np.pi*v) + 0.7

	result = 1 - element2

	return result

def main(job_id, params):
	return mix_cosines(params['x1'],params['x2'])