import numpy as np
import math

def drop_wave(x1,x2):
	numerator = 1 + np.cos(12*((x1**2 + x2**2)**0.5))
	delimiter = 0.5 * (x1**2 + x2**2) + 2

	result = - numerator / delimiter

	return result

def main(job_id, params):
	return drop_wave(params['x1'], params['x2'])