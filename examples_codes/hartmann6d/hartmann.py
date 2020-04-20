import numpy as np
import math

def hartmann(x1,x2,x3,x4,x5,x6):

	A = [[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]]

	A = np.array(A)

	P = [[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],[2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]]

	P = np.array(P)
	P = 1e-4*P

	alpha = [1.0,1.2,3.0,3.2]

	expVal = np.exp((-(A[:,0]*(x1-P[:,0])**2 + A[:,1]*(x2-P[:,1])**2 + A[:,2]*(x3-P[:,2])**2 + A[:,3]*(x4-P[:,3])**2 + A[:,4]*(x5-P[:,4])**2 + A[:,5]*(x6-P[:,5])**2)))
	y = -(alpha[0]*expVal[0] + alpha[1]*expVal[1] + alpha[2]*expVal[2] + alpha[3]*expVal[3])


	print 'Result = %f' %y
	return y

def main(job_id, params):
	return hartmann(params['x1'],params['x2'],params['x3'],params['x4'],params['x5'],params['x6'])