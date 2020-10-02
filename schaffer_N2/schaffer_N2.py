import numpy as np
import math

def schaffer_N2(x1, x2):
	numerator = (np.sin(x1**2 - x2**2))**0.5 - 0.5
	delimeter = (1 + 0.001*(x1**2 + x2**2)) ** 2

	result = 0.5 + numerator / delimeter

	return result

def main(job_id, params):
	return schaffer_N2(params['x1'], params['x2'])
