import numpy as np
import math

def styblinski_tang_8d(x1,x2,x3,x4,x5,x6,x7,x8):
	Xs = [x1,x2,x3,x4,x5,x6,x7,x8]

	result = 0

	for dim in range(len(Xs)):
		x = Xs[dim]
		d = dim + 1

		first = x**4
		second = 16 * (x**2)
		third = 5 * x

		prod = first - second + third

		result = result + prod

	return 0.5 * result

def main(job_id, params):
	return styblinski_tang_8d(params['x1'],params['x2'],params['x3'],params['x4'],
		params['x5'],params['x6'],params['x7'],params['x8'])
