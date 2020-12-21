import numpy as np
import math
import matlab.engine
import matlab
from scipy import io

def mse_rcwa(x1,x2,x3,x4,x5, case=0):
	
	eng = matlab.engine.start_matlab()

	X = np.array((x1,x2,x3,x4,x5))
	S_spec = np.arange(300,801,5).reshape(-1)

	simulated_spec = eng.S_rcwa_R(matlab.double( X.tolist() ), matlab.double( S_spec.tolist() ), nargout = 1)
	simulated_spec = np.array(simulated_spec).reshape(-1)

	mat_file = io.loadmat("Spec_CD.mat")

	O_spec = mat_file["GN_spec"]

	R_spec = O_spec[:,:,case]

	R_spec1 = R_spec[:,1]
	R_spec2 = R_spec[:,2]
	R_spec3 = R_spec[:,3]

	R_spec_total = np.array([R_spec1, R_spec2, R_spec3]).reshape(-1)
	R_spec_total = R_spec_total.real

	result = sum((simulated_spec - R_spec_total)**2) / 303

	return result

def main(job_id, params):
	return mse_rcwa(params['x1'], params['x2'], params['x3'], params['x4'],params['x5'])
