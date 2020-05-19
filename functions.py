import numpy as np
import scipy.stats as sps
import pymongo
from spearmint.utils import compression


## getting sampled_x_stars in a mongodb document
def get_x_stars(jobs_exp):
	x_stars_exp = []
	for experiment_num in range(len(jobs_exp)):
		cur_docs = jobs_exp[experiment_num]
		sample_x_star_iter = []
		for doc in cur_docs.find():
			x_stars_dict = compression.decompress_nested_container(doc['x star samples'])
			x_stars_kernels = []

			for kernel in x_stars_dict.keys():
				for x_star in x_stars_dict[kernel]:
					x_stars_kernels.append(x_star)

			sample_x_star_iter.append(x_stars_kernels)
		x_stars_exp.append(samples_x_star_iter)

	return x_stars_exp


## Getting conditioned Variances in suggested input location

def get_cond_vars(jobs_exp):
    cond_vars_exp = []
    for exp_num in range(len(jobs_exp)):
        cur_docs = jobs_exp[exp_num]
        cond_vars_iter = []
        for doc in cur_docs.find():
            cond_vars_dict = compression.decompress_nested_container(doc['cond_Vars'])
            for key in cond_vars_dict.keys():
                cond_vars = cond_vars_dict[key]
                cond_vars_iter.append(cond_vars[0]["Objective"])
            
        cond_vars_exp.append(cond_vars_iter)
    return cond_vars_exp


## Getting unconditioned Variances in suggested input location

def get_uncond_vars(jobs_exp):
    uncond_vars_exp = []
    for exp_num in range(len(jobs_exp)):
        cur_docs = jobs_exp[exp_num]
        uncond_vars_iter = []
        for doc in cur_docs.find():
            uncond_vars_dict = compression.decompress_nested_container(doc['uncond_Vars'])
            for key in uncond_vars_dict.keys():
                uncond_vars = uncond_vars_dict[key]
                uncond_vars_iter.append(uncond_vars[0]["Objective"])
            
        uncond_vars_exp.append(uncond_vars_iter)
    return uncond_vars_exp


## Getting acquisition function value in suggested input location

def get_acq_value(jobs_exp):
    acq_value_exp = []
    for exp_num in range(len(jobs_exp)):
        cur_docs = jobs_exp[exp_num]
        acq_value_iter = []
        for doc in cur_docs.find():
            acq_value = doc['acq_value']
            acq_value_iter.append(acq_value)
        acq_value_exp.append(acq_value_iter)
    return acq_value_exp


## Getting objective values in optimization process

def get_objective(recomm_exp):
    objs_exp = []
    for exp_num in range(len(recomm_exp)):
        cur_docs = recomm_exp[exp_num]
        objs_iter = []
        for doc in cur_docs.find():
            objs_iter.append(compression.decompress_nested_container(doc['objective']))
        objs_exp.append(objs_iter)
    return objs_exp



## Getting suggested input locations in optimization process

def get_suggested_locs(jobs_exp):
    suggested_locs_exp = []
    for exp_num in range(len(jobs_exp)):
        cur_docs = jobs_exp[exp_num]
        locs_iter = []
        for doc in cur_docs.find():
            locs_iter.append(vectorify(compression.decompress_nested_container(doc['params'])))
        suggested_locs_exp.append(locs_iter)
    return suggested_locs_exp

## Getting recommedated input locations in optimization proces

def get_recomm(recomm_exp):
    recomms_exp = []

    for exp_num in range(len(recomm_exp)):
        cur_docs = recomm_exp[exp_num]
        recomm_iter = []
        for doc in cur_docs.find():
            recomm_param = compression.decompress_nested_container(doc['params'])
            recomm_iter.append(vectorify(recomm_param))
        recomms_exp.append(recomm_iter)
    return recomms_exp

## Converts a single input params to the corresponding vector
def vectorify(params):
    v = []

    for name, param in sorted(params.iteritems()):
        v.append(param['values'][0])

    return v
