    def compute_acquisition_function_CE(self, acquisition_function, grid, tasks, fast_update):

        # at first we have only Mk ep sols and x_stars
        # if wanna get another sample of hyperaparameter, should compute the x_stars and Ep sols 
        # using performEPandgettingXstar method.......

        logging.info("Computing %s using Cross Entropy for %s. "% (self.acquisition_function_name, ', '.join(tasks)))


        avg_hypers = function_over_hypers


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% using CE algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # 1. Initialize : sample Mk hyperparameter and GP model fitting already done......
        #               : Initialize pdf p(-, v1)
        #               : i = 1
        #    while( stopping criteria is not satisfied ):
        #        2. Generation : randomly generate a set of solutions X1,X2...,X_kt from pdf p(.,v_t)
        #        3. Simulation : simulate Ni replications for each Xi to get objective function
        #                      : that is, fitting Ni hyperparameter for GP model
        #                      : Get acquisition function values from that GP model
        #                      : Compute sample averaged acquistion function value 
        #        4. Update     : Update pdf parameter v_hat_t+1 
        #                      : from equation; v_hat_t+1 = argmax ΣI{if solution i is elite}logp(.,v_t)
        #                      : v_t+1 = α*v_hat_t+1 + (1-α)*v_t
        #                      t = t+1
        #    
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # Initializing parameters 
        t = 1
        k = 500
        T = 2500
        m = int(k * 0.1)
        alpha = 0.5
        N = int(T/k)
        sum_T = T

        # fitting n0 hyperparameters is already done.....
        # getting k solutions from uniform distribution
        candidates = np.random.uniform(low=0, high=1, size=(k,self.num_dims))
        
        # Current, sampling from truncated multivariate normal distribution is a bottleneck i think......
        # Right now, do it through "Acceptance-Rejection" method.......

        # Constructing elite set by top m solutions.....
        # Getting acquisition function values in current GP model
        acq_candidates = avg_hypers(self.models.values(), acquisition_function, 
                                                   candidates, compute_grad = False, tasks = tasks)
        

        copied = copy.deepcopy(acq_candidates)
        #logging.info(copied)
        #logging.info(len(copied))
        copied = np.sort(copied)[::-1]


        elite_set = []
        for i in range(m):
            origin_index = np.where(acq_candidates==copied[i])
            elite_set.append(candidates[origin_index][0])


        #logging.info(elite_set)


        mu = np.average(elite_set, axis= 0)
        mu = np.reshape(mu, (len(mu),1))
        

        zero_vec = np.reshape([0,0], (len([0,0]),1))
        cov_mat = np.matmul(zero_vec,zero_vec.T)
        
        cov_mat = np.matrix(cov_mat)
        
        for item in elite_set:
        	cov_mat = cov_mat + np.matrix(np.matmul((np.reshape(item,(len(item),1))),(np.reshape(item,(len(item),1))).T))

        cov_mat = cov_mat / m

        v = (mu,cov_mat)


        stopping_criteria = (sum_T >= 200e4)
        sum_N = 5
        while(not stopping_criteria):
            candidates = []
            k = int(1.04 * k)
            T = int(1.1 * T)
            m = int(0.1 * k)

            sum_T += T
            N = 1
            


            # Generating solutions process
            while(True):
                # one sample from pdf
                #logging.info(v[0].reshape(self.num_dims,1))
                #logging.info(type(v[1]))
                x = np.random.multivariate_normal(v[0].reshape(self.num_dims),v[1])
                # if there is any value which is not in search space, reject that sample and sample again
                # else; append it candidates
                out_range = False
                for i in range(self.num_dims):
                    if(x[i]<0 or x[i]>1):
                        out_range = True
                        continue;

                if(out_range):
                    continue;

                candidates.append(x)

                if(len(candidates) >= m):
                    break;


            #logging.info(candidates)
            # Fitting N hyperparameter and evaluate acquisition function value at location of candidates

            for i in range(N):
            	sum_N += 1
            	#logging.info(sum_N)
            	#logging.info("Fitting %d the hyperparameter and getting acquisition function value"%i)
                for task_name in tasks:
                    gp_instance_task = self.models[task_name]

                    inputs = gp_instance_task._inputs
                    values = gp_instance_task._values

                    gp_instance_task.fit_another(inputs,values)


                # perform EP and X star sampling in current state
                self.acquisition_function_instance.performEPandXstarSamplingForOneState(self.objective_model_dict.values()[0],
                                                           self.constraint_models_dict, fast_update,
                                                           num_random_features = self.options['pes_num_rand_features'],
                                                           x_star_tolerance = self.options['pes_opt_x*_tol'],
                                                           num_x_star_samples = self.options['pes_num_x*_samples'] )

                # compute acquisition function of current state

                # Same with PES.py "acquisition" function


                inputs_hash = hash(np.array(candidates).tostring())

                cache_key = (inputs_hash, self.objective_model_dict.values()[0].state)

                if cache_key in self.acquisition_function_instance.stored_acq:
                    return sum([self.acquisition_function_instance.stored_acq[cahce_key][task] for task in tasks])

                N_cand = np.array(candidates).shape[0]

                x_stars = self.acquisition_function_instance.cached_x_star[self.objective_model_dict.values()[0].state]
                ep_sols = self.acquisition_function_instance.cached_EP_solutions[self.objective_model_dict.values()[0].state]

                temp_acq_dict = defaultdict(lambda : np.zeros(N_cand))

                for i in xrange(self.options['pes_num_x*_samples']):

                    if x_stars[i] is None:
                        continue

                    for t,val in pes.evaluate_acquisition_function_given_EP_solution(self.objective_model_dict,self.constraint_models_dict, np.array(candidates), ep_sols[i], x_stars[i]).iteritems():
                        temp_acq_dict[t] += val

                for t in temp_acq_dict:
                    temp_acq_dict[t] = temp_acq_dict[t] / float(self.options['pes_num_x*_samples'])


                self.acquisition_function_instance.stored_acq[cache_key] = temp_acq_dict


                if tasks is None:
                    tasks = temp_acq_dict_keys()


            # We got sample averaged acquisition function values at location of candidates
            newly_acq_candidates = avg_hypers(self.models.values(), acquisition_function, 
                                                         np.array(candidates), compute_grad = False, tasks = tasks)


            copied = copy.deepcopy(newly_acq_candidates)
            copied = np.sort(copied)[::-1]

            elite_set = []

            for i in range(m):
                origin_index = np.where(newly_acq_candidates==copied[i])[0][0]
                
                elite_set.append(candidates[origin_index])


            prev_v = v

            #logging.info(elite_set)
            mu = np.average(elite_set, axis = 0)
            mu = np.reshape(mu, (len(mu),1))


            zero_vec = np.reshape([0,0], (len([0,0]),1))
            cov_mat = np.matmul(zero_vec,zero_vec.T)
            cov_mat = np.matrix(cov_mat)

            for item in elite_set:
            	cov_mat = cov_mat + np.matrix(np.matmul((np.reshape(item,(len(item),1))),(np.reshape(item,(len(item),1))).T))

            cov_mat = cov_mat / m

            
            #logging.info("mu is %s"%str(mu))
            #logging.info("cov is %s"%str(cov_mat))
            v = (mu, cov_mat)


            stopping_criteria = (sum_T >= 200e4) or (sum_N >= 20)
            #logging.info(sum_N)

            # while loop in ended in here

        mu = np.reshape(mu,len(mu))
        
        
        best_acq_value = avg_hypers(self.models.values(), acquisition_function, 
                                                np.array(mu), compute_grad = False, tasks = tasks)[0]

        logging.info("location is %s"%str(mu))
        logging.info("value is %s"%str(best_acq_value))
        return {"location" : mu, "value": best_acq_value}
