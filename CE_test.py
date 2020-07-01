import numpy as np
import numpy.random as npr
import scipy.stats as sps
import numpy.linalg as npla
import scipy.optimize as spo
import pdb
import copy 

import logging

def cross_entropy(function, dimension, options, total_Simul_Budget = 400e4):

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CE algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# 1. Initialize 
	#    first uninformative prior probability density distribution pdf : uniform dist'n
	#    Randomly generate k solutions from that distribution
	# 2. Simulation 
	#    Simulate Ni replications for each solution to get objective function value 
	#    Get S^(X) for each solution in all candidates
	# 3. Elite Set and Update
	#    Construct Elite set and update the parameter vector using elite set solutions......
	# 4. Stop
	#    If stopping criteria is satisfied, stop
	#    If not, go to the generation stage......
	#

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# First Initialize process

	# parameters
	k= 500       # first randomly generating solution size
	T = 2500     # first Simulating Budget
	m = int(k*0.1)    # first Elite set size
	alpha = 0.5  # First v parameter learning rate
	N = T/k      # first simulating budget for each solution 
	   # Used Simulation Budget 
	threshold = None
	v = None

	# Uninformative uniform density
	#init_mean = [0.5,0.5]
	#init_cov = [[5,0],[0,5]]

	# n dimension vector generating
	init_mean = []
	for i in range(dimension):
		init_mean.append(0.5)

	init_cov = []
	for i in range(dimension):
		temp_vec = []
		for j in range(dimension):
			if(i==j):
				temp_vec.append(5)
				continue
			temp_vec.append(0)
		init_cov.append(temp_vec)

	candidates = []
	while(True):
		x = np.random.multivariate_normal(init_mean,init_cov)

		out_range = False
		for i in range(dimension):
			if(x[i]>0 or x[i]>1):
				out_range = True
				break

		if(out_range):
			continue

		candidates.append(x)

		if(len(candidates)>= k):
			break;


	#print candidates

	# mapping process is needed in here...........####################################

	mapped_cand = []
	for item in candidates:

		temp_item = []

		for dim in range(len(item)):
			# from options getting search space
			# item[dim] * (max-min) + min
			temp_item.append(item[dim] * (options['x%d'%(dim+1)]['max'] - options['x%d'%(dim+1)]['min'])+options['x%d'%(dim+1)]['min'])

		temp_item = np.array(temp_item)
		mapped_cand.append(temp_item)


	###################################################################################

	# Getting Ni function values from pdf
	func_vals = []
	for solution in mapped_cand:

		avg_est_val = 0

		for i in range(N):
			avg_est_val = avg_est_val + function(*solution)

		func_vals.append(avg_est_val/N)


	sum_T = T 

	elite_set = []
	copied = copy.deepcopy(func_vals)
	copied.sort()

	for i in range(m):
		origin_index = func_vals.index(copied[i])
		elite_set.append(candidates[origin_index])

	


	
	# Updating parameter and generating process
	while(True):
		print("Updating process is being processed")
		# updating parameter

		mean = np.mean(elite_set, axis=0)

		zero_vec = np.reshape(np.zeros(dimension), (dimension,1))
		cov_mat = np.matrix(np.matmul(zero_vec,zero_vec.T))

		for item in elite_set:
			cov_mat = cov_mat + np.matrix(np.matmul(np.reshape(item,(dimension,1)),np.reshape(item,(dimension,1)).T))

		cov_mat = cov_mat / m

		prev_v = v
		v = (mean,cov_mat)
		prev_threshold = threshold
		print(m)
		print(len(copied))
		print(sum_T)
		threshold = (copied[m] + copied[m-1])/2

		stopping_criteria = (sum_T >= total_Simul_Budget) #or (prev_v == v) or (threshold == prev_threshold)
		# stopping criteria if break.........
		if(stopping_criteria):
			print(sum_T)
			break


		k = int(1.04 * k)
		T = int(1.1  * T)
		m = int(0.1  * k)
		N = int(T/k)

		# Updating Process is done

		# Generation and Simulation process 
		candidates = []
		while(True):
			x = np.random.multivariate_normal(mean,cov_mat)

			# if there is any value which is not in search space, reject that sample 
			# and sample again
			out_range = False
			for i in range(dimension):
				if(x[i]<0 or x[i]>1):
					out_range = True
					break

			if(out_range):
				continue

			candidates.append(x)

			if(len(candidates) >= k):
				break;

		# Solution Generation Process is done

		# mapping process is needed in here......###########################################
		mapped_cand = []
		for item in candidates:
			temp_item = []
			for dim in range(len(item)):
				# from options getting search space
				# item[dim] * (max-min) + min
				temp_item.append(item[dim] * (options['x%d'%(dim+1)]['max'] - options['x%d'%(dim+1)]['min'])+options['x%d'%(dim+1)]['min'])
			temp_item = np.array(temp_item)
			mapped_cand.append(temp_item)



		#####################################################################################

		# Constructing elite_set

		func_vals = []
		for solution in mapped_cand:
			avg_est_val = 0

			for i in range(N):
				avg_est_val = avg_est_val + function(*solution)

			func_vals.append(avg_est_val/N)

		sum_T = sum_T + T


		elite_set = []
		copied = copy.deepcopy(func_vals)
		copied.sort()

		for i in range(m):
			origin_index = func_vals.index(copied[i])
			elite_set.append(candidates[origin_index])


		print("current mean is %s"%str(mean))
		# end of while


	x_opt = []
	for dim in range(len(mean)):
		x_opt.append(mean[dim] * (options['x%d'%(dim+1)]['max']-options['x%d'%(dim+1)]['min']) + options['x%d'%(dim+1)]['min'])





	return x_opt