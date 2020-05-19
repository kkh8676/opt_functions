import numpy as np
import math

def levy_N13(x1,x2):
	a = np.sin(3*np.pi*x1)**2
	b = ((x1-1)**2)*(1+(np.sin(3*np.pi*x2)**2))
	c = ((x2-1)**2)*(1+(np.sin(2*np.pi*x2)**2))

	result = a + b + c

	return result

def main(job_id, params):
	return levy_N13(params['x1'], params['x2'])