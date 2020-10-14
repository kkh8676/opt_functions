import numpy as np
import math

def michalewicz_2d(x1,x2):
	result = 0

	Xs = [x1,x2]

	
	for dim in range(len(Xs)):
		# dim is 0 to len-1
		x = Xs[dim]
		d = dim+1

		first_term = np.sin(x)
		second_term = np.sin(d*(x**2)/np.pi) ** (2*10)

		prod = first_term * second_term
		result = result + prod 

	result = -result

	return result

def main(job_id, params):
	return michalewicz_2d(params['x1'],params['x2'])