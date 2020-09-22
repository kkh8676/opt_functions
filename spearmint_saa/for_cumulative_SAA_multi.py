    # variance of the optimality gap should be added in the optimimality gap
    
    
    # 20.09.22.
    # modified SAA, using cumulative hyperparameter samples......
    # cumulative SAA + multistart, PSO
    def suggest_v3(self, task_couplings):

        if not isinstance(task_couplings, dict):
            task_couplings = {task_name : 0 for task_name in task_couplings}

        task_names = task_couplings.keys()

        # Indeed it does not make sense to compute the best() and all that if we
        # have absolutely no data. 
        # But I want to be careful here because of the problem I had before, that we
        # never observed the objective (kept getting NaNs) and so kept picking randomly
        # that is not good at all-- need to use the GPs to model the NaNs.
        # so, if I REALLY have no data, then I want to do this. But that means no data
        # from ANY of the tasks. 
        if self.total_inputs < self.options['initial_design_size']:
            # design_index = npr.randint(0, grid.shape[0])
            # suggestion = self.input_space.from_unit(grid[design_index])
            total_pending = sum(map(lambda t: t.pending.shape[0], self.tasks.values()))
            # i use pending as the grid seed so that you don't suggest the same thing over and over
            # when you have multiple cores -- cool. this was probably weird on the 3 core thing
            suggestion = sobol_grid.generate(self.num_dims, grid_size=100, grid_seed=total_pending)[0]
            suggestion = np.random.rand(self.num_dims)
            # above: for some reason you can't generate a grid of size 1. heh.

            suggestion = self.input_space.from_unit(suggestion) # convert to original space

            logging.info("\nSuggestion:     ")
            self.input_space.paramify_and_print(suggestion.flatten(), left_indent=16)
            if len(set(task_couplings.values())) > 1: # if decoupled
                # we need to pick the objective here
                # normally it doesn't really matter, but in the decoupled case
                # with PESC in particlar, if there are no objective data it skips
                # the EP and gets all messed up
                return suggestion, [self.objective.name]
                # return suggestion, [random.choice(task_names)]
            else:  # if not decoupled. this is a bit of a hack but w/e
                return suggestion, task_names

        fast_update = False
        if self.options['fast_updates']:
            fast_update = self.start_time_of_last_slow_update <= self.end_time_of_last_slow_update
            # this is FALSE only if fit() set self.start_time_of_last_slow_update,
            # indicating that we in the the process of a slow update. else do a fast update
        self.fast_update = fast_update

        # Compute the current best if it hasn't already been computed by the caller
        # and we want to make recommendations at every iteration

        # 20.08.05
        # so add the condition if we are using EI, we can jump this process, not done yet
        if not self.best_computed and self.options['recommendations'] == "during" and self.options["acquisition"] == 'EI':
            self.best() # sets self.stored_recommendation
        # only need to do this because EI uses current_best_value --
        # otherwise could run the whole thing without making recommendations at each iteration
        # hmm... so maybe make it possible to do this when using PESC
        if self.options['recommendations'] == "during":
            current_best_value = self.stored_recommendation['model_model_value']
            if current_best_value is not None:
                current_best_value = self.objective.standardize_variance(self.objective.standardize_mean(current_best_value))
        else:
            current_best_value = None

        ## can jump code above


        # Create the grid of optimization initializers
        acq_grid = self.generate_grid(self.options['fast_acq_grid_size'] if fast_update else self.options['acq_grid_size'])

        # flip the data structure of task couplings
        task_groups = defaultdict(list)
        for task_name, group in task_couplings.iteritems():
            task_groups[group].append(task_name)
        # ok, now task_groups is a dict with keys being the group number and the
        # values being lists of tasks

        # note: PESC could just return
        # the dict for tasks and the summing could happen out here, but that's ok
        # since not all acquisition functions might be able to do that
        task_acqs = dict()


        
        # SAA algorithm, referencing SAA algorithm
        # 1. Initial Sample size N, N'
        #    decision rule for determining M
        #    increasing N,N' rule, tolerance ε
        #
        # 2. For m-1,2,3,.....M
        #        2.1 Generate a sample size of N
        #            Solve the problem minimizing N sample average of G
        #        2.2 Compute g_N'_(x*_N_m)
        #            Compare with g_N'(x*_N_m'), m'<m
        #            x* <- good one x*_N_m or x*_N_m'
        # 
        # 3. Estimate Optimality Gap g_hat(x_hat) - v*
        #    Compute the variance of that estimator 
        #    Estimation method 1 ) g_hat_N'(x_hat) - v_bar^M_N ; g_hat_N'(x_hat) = 1/N' * (sum of G(x_hat, W_n))
        #    Estimation method 2 ) g_bar^M_N(x_hat) - v_bar^M_N; g_bar^M_N(x_hat) = 1/M * (sum of g_hat^m_N(x_hat))
        # 
        # 4. Optimality Gap is too large, increase N or N' and return to 2
        #    otherwise, choose the best solution among candidates. x_hat^m_N , m=1,2,....,M
        #    through screening and selection procedure using the information we got so far.

        # 1. Initialization process
        sample_size_N = 1
        replication_M = 25
        sample_size_N_prime = int(replication_M * sample_size_N * 0.8)

        relative_gap_thres = 0.15  # this variable is needed??

        N_increasing_num = 1 # increasing Rule of N,N'

        #z_alpha = 

        round_num = 0

        while(True):
            # 2. sample average approximation optimization for M replication

            # fit hyperparameter 

            if(round_num == 0):
                for task_name, task in self.tasks.iteritems():
                    self.models[task_name].fit_start(
                        self.models[task_name]._inputs,
                        self.models[task_name]._values,
                        pending = task.normalized_pending(self.input_space),
                        hypers=None,
                        fit_hypers = True,
                        increasing_num = replication_M * sample_size_N)
            else:
                for task_name, task in self.tasks.iteritems():
                    self.models[task_name].fit_incre(
                        self.models[task_name]._inputs,
                        self.models[task_name]._values,
                        pending = task.normalized_pending(self.input_space),
                        hypers = None,
                        fit_hypers = True,
                        increasing_num = replication_M * sample_size_N - self.models[task_name].num_states)

            # initialize/create the acquisition function
            logging.info("Initializing %s in round %d" %( self.acquisition_function_name, round_num))
            
            acquisition_function = self.acquisition_function_instance.create_acquisition_function(\
                self.objective_model_dict, self.constraint_models_dict,
                fast=fast_update, grid=acq_grid, current_best=current_best_value,
                num_random_features=self.options['pes_num_rand_features'],
                x_star_grid_size=self.options['pes_x*_grid_size'], 
                x_star_tolerance=self.options['pes_opt_x*_tol'],
                num_x_star_samples=self.options['pes_num_x*_samples'])


            # in create_acquisition_function, we do performEPandXstarSamplingForOneState process 
            # below code is not neeeded 


            # function_over_hypers(self.models.values(), self.acquisition_function_instance.performEPandXstarSamplingForOneState,
            #         self.objective_model_dict.values()[0],
            #         self.constraint_models_dict,
            #         fast_update,
            #         self.options['pes_num_rand_features'],
            #         self.options['pes_opt_x*_tol'],
            #         self.options['pes_num_x*_samples'])

            acq_opt_logs = []
            # for replication M
            for m in range(replication_M):
                

                # performEP and X star sampling for those newly sampled hyperparameters
                # in performEP and X star sampling method in PES.py 
                # there are process which jump if there is existing EP sol and x star sol
                # so in this process, function_over_hypers process don't need to be modified



                # get the optimizing solution for sampled average approximation of those hyperparameters
                # compute_acquisition_function_for_newly_sampled_hyper should be coded
                # because this sample average path is from newly sampled hyperparameter not total.
                for group, task_group in task_groups.iteritems():
                    # def compute_acquisition_function_for_newly_sampled_hyper(self, acquisition_function, grid, tasks, fast_update):
                    
                    task_acqs[group] = self.compute_acquisition_function_for_specific_hyper(acquisition_function, m, sample_size_N, acq_grid, task_group, fast_update)
                    # this compute_acquisiton_function should be modified 
                    # because that function perform averaging process for all states of GP instance
                    # in this algorithm, we should average of acquisition functions from newly sampled hyperparameter
                    
                    
                    acq_opt_logs.append(task_acqs[group])


                # task_groups : decoupled task groups
                # group : number of decoupled task groups, dictionary key
                # task_group : in the number of group, objective ans const1 const2 and so on....

                # task acqs in a dict, with keys being the arbitrary group index, and the values
                # being a dict with keys "location" and "values"

            # now we have M solutions in the acq_opt_logs 
            # acq_opt_logs got length M


            # 2.2 Estimate optimality gap 
            #     Compute the variances of that estimator
            # v_bar^M_N can be computed using acq_opt_log[]["value"]
            # estimator of 'g' part is little bit complicated............number1 and number2 
            #logging.info([dictionary["value"] for dictionary in acq_opt_logs])


            # x_hat should be chosen in acq_opt_logs by the performance of g_N_prime
            # using compute_acquisition_function, deleting if self.options["optimize_acq"] part
            # and parameter 'grid' gets the list of locations in 'acq_opt_logs'
            x_hat_cands = np.array([ dictionary["location"] for dictionary in acq_opt_logs ])
            
            N_prime_acq_opt = dict()
            for group, task_group in task_groups.iteritems():
                # %%%%%%%%%% should be modified %%%%%%%%
                # randomly get N_prime number of states in the GP 
                # and compute acquisition_ function for that states.......

                # x_hat should be selected from x_hat_cands....
                # so, in compute_acquisition_function_subset no optimize acq part.., just calculate the function and get the best in x_hat_cands
                # def compute_acquisition_function_subset(self, acquisition_function, grid, num_subset, tasks, fast_update):
                N_prime_acq_opt[group] = self.compute_acquisition_function_subset(acquisition_function, x_hat_cands, sample_size_N_prime,
                                                        task_group, fast_update)

            # x_ hat is the best location in N_prime_acq_opt["location"]
            # which have best performance in g_N_prime

            # this code should be modified for constrained version
            #logging.info(N_prime_acq_opt)
            x_hat = N_prime_acq_opt[group]["location"]
            
            #logging.info("x hat is %s"%str(x_hat))
            #logging.info("x_hat is %s"%str(np.array([x_hat])))
            # using N prime number of sample averaging .........
            # for that purpose, function_over_hypers method should be modified
            # should get 'function' ?
            # 'grid' parameter gets the value of x_hat
            # this value is N_prime_acq_opt["value"]
            g_N_prime_at_x_hat = N_prime_acq_opt[group]["value"]

            # using total number of sample averaging
            # for this purpose, function_over_hypers don't need to be modified..
            g_bar_N_M_at_x_hat = function_over_hypers(self.models.values(),acquisition_function,
                np.array([x_hat]), compute_grad = False, tasks = task_group)

            est_v_star = np.average([dictionary["value"] for dictionary in acq_opt_logs])

            

            # optimality gap 
            # g_N_prime(x_hat) - est_v_star ; number 1
            # g_bar_N_M(x_hat) - est_v_star ; number 2 

            # need to be modified for variance term
            opt_gap_v1 = np.abs(g_N_prime_at_x_hat - est_v_star)
            opt_gap_v2 = np.abs(g_bar_N_M_at_x_hat - est_v_star)

            # We need to know which one is the larger!!!
            if opt_gap_v1 < opt_gap_v2:
                logging.info("using N prime gap is smaller!!")
            else:
                logging.info("using total N*M gap is smaller!!")

            optimality_gap = opt_gap_v1 if opt_gap_v1 < opt_gap_v2 else opt_gap_v2

            stopping_criteria = optimality_gap < np.abs(relative_gap_thres * est_v_star)
            
            # logging.info("optimality gap is %s"%str(optimality_gap))
            # logging.info("g_N_prime_at_x_hat is %s"%str(g_N_prime_at_x_hat))
            # logging.info("g_bar_N_M_at_x_hat is %s"%str(g_bar_N_M_at_x_hat))
            # logging.info("est v star is %s"%str(est_v_star))
            # logging.info("threshold value is %s"%str(np.abs(relative_gap_thres * est_v_star)))
            # logging.info(" ")
            if(stopping_criteria):
                # break the while loop
                # Maybe multistart algorithm can be added in here...........
                # multistart version uses compute_acquisition_function method 
                # but in SAA boundary, we should use it in 'suggest' method........
                # we have 'x_hat_cands' which are the starters of multistart and PSO
                final_task_acqs = defaultdict(list)
                for group, task_group in task_groups.iteritems():
                    final_task_acqs[group] = self.compute_acquisition_function_multistart(acquisition_function, acq_grid, x_hat_cands, task_group, fast_update)

                break


            # if stopping criteria is not satisfied........
            # increase N and round num ........we can track the previous information using round num and N
            # right now increasing Rule for N and N' is heuristically
            sample_size_N = sample_size_N + N_increasing_num
            sample_size_N_prime = int(replication_M * sample_size_N * 0.8) # modified to fraction
            round_num = round_num + 1
            acq_opt_logs = []

            # we need to clean the GP instances state.......
            # gp_instance._cache_list 
            # _hypers_list
            # self.state
            # num_states
            # in 'acquisition' method, there is code which returns existing acq_values 


            # so just fitting the new GP model with newly sampled hyperparameters

        # after while loop is over
        # in task_acqs[group] last solution will be saved....... i think

        # screening and selection procedure for solutions in acq_opt_logs.......%%
        

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # normalize things by the costs
        group_costs = dict()
        for task_name, group in task_couplings.iteritems():
            # scale duration is false by default setting
            if self.options['scale_duration']:

                # scale the acquisition function by the expected duration of the task
                # i.e. set the cost to the expected duation
                expected_duration = np.exp(self.duration_models[task_name].predict(task_acqs[group]["location"][None])[0]) # [0] to grab mean only

                # now there are 2 cases, depending on whether you are doing the fast/slow updates
                if self.options['fast_updates']:
                    # complicated case
                    # try to predict whether the next update will be slow or fast...

                    if self.options['predict_fast_updates']:

                        # fast_update --> what we are currently doing
                        if fast_update: # if currently in a fast update
                            predict_next_update_fast = (time.time() - self.end_time_of_last_slow_update + expected_duration)*self.options['thoughtfulness'] < self.duration_of_last_slow_update 
                        else: # we are in a slow update
                            predict_next_update_fast = expected_duration*self.options['thoughtfulness'] < self.duration_of_last_slow_update

                        if predict_next_update_fast:
                            # predict fast update
                            # have we done a fast update yet:
                            logging.debug('Predicting fast update next.')
                            if self.duration_of_last_fast_update > 0:
                                expected_thinking_time = self.duration_of_last_fast_update
                            else:
                                expected_thinking_time = 0.0
                            # otherwise don't add anything
                        else:
                            logging.debug('Predicting slow update next.')
                            if self.duration_of_last_slow_update > 0:
                                expected_thinking_time = self.duration_of_last_slow_update
                            else:
                                expected_thinking_time = 0.0

                    else: # not predicting -- decided to use FAST time for this, now SLOW time!
                        if self.duration_of_last_fast_update > 0:
                            expected_thinking_time = self.duration_of_last_fast_update
                        else:
                            expected_thinking_time = 0.0                    

                else: # simpler case
                    if self.duration_of_last_slow_update > 0:
                        expected_thinking_time = self.duration_of_last_slow_update
                    else:
                        expected_thinking_time = 0.0
                
                expected_total_time = expected_duration + expected_thinking_time # take the job time + the bayes opt time

                logging.debug('   Expected job duration for %s: %f' % (task_name, expected_duration))
                logging.debug("   Expected thinking time:  %f" % expected_thinking_time)
                logging.debug('   Total expected duration: %f' % expected_total_time)
                # we take the exp because we model the log durations. this prevents us
                # from ever predicting a negative duration...
                # print '%s: cost %f' % (task_name, group_costs[group])
                group_costs[group] = expected_total_time
            else:
                group_costs[group] = self.tasks[task_name].options["cost"]

        # This is where tasks compete
        if len(task_groups.keys()) > 1: # if there is competitive decoupling, do this -- it would be fine anyway, but i don't want it to print stuff
            for group, best_acq in N_prime_acq_opt.iteritems():
                best_acq["value"] /= group_costs[group]
                if group_costs[group] != 1:
                    logging.debug("Scaling best acq for %s by a %s factor of 1/%f, from %f to %f" % ( \
                            ",".join(task_groups[group]), 
                                "duration" if self.options['scale_duration'] else "cost",
                            group_costs[group],
                            best_acq["value"]*group_costs[group],
                            task_acqs[group]["value"]))
                else:
                    logging.debug("Best acq for %s: %f" % (task_groups[group], task_acqs[group]["value"]))

        # Now, we need to find the task with the max acq value
        max_acq_value = -np.inf
        best_group = None
        for group, best_acq in final_task_acqs.iteritems():
            if best_acq["value"] > max_acq_value:
                best_group = group
                max_acq_value = best_acq["value"]

        # Now we know which group to evaluate
        suggested_location = final_task_acqs[best_group]["location"]
        best_acq_value     = final_task_acqs[best_group]["value"]
        suggested_tasks    = task_groups[best_group]

        # Make sure we didn't do anything weird with the bounds
        suggested_location[suggested_location > 1] = 1.0
        suggested_location[suggested_location < 0] = 0.0

        suggested_location = self.input_space.from_unit(suggested_location)

        logging.info("\nSuggestion: task(s) %s at location" % ",".join(suggested_tasks))
        self.input_space.paramify_and_print(suggested_location.flatten(), left_indent=16)

        if not fast_update:
            self.end_time_of_last_slow_update = time.time() # the only one you need for fast/slow update, the rest for scale-duration
            self.duration_of_last_slow_update = time.time() - self.start_time_of_last_slow_update
        else:
            self.end_time_of_last_fast_update = time.time()
            self.duration_of_last_fast_update = time.time() - self.start_time_of_last_fast_update

        return suggested_location, suggested_tasks
        
        
    # 20.09.23
    # using multistart........in SAA
    def compute_acquisition_function_multistart(self, acquisition_function, grid, x_hat_cands, tasks, fast_update):

        logging.info("Computing %s on grid for %s." % (self.acquisition_function_name, ', '.join(tasks)))


        # Special case -- later generalize this to more complicated cases
        # If there is only one task here, and it depends on only a subset of the parameters
        # then let's do the optimization in lower dimensional space... right?
        # i wonder though, will the acquisition function actually have 0 gradients in those
        # directions...? maybe not. but they are irrelevant. but will they affect the optimization?
        # hmm-- seems not worth the trouble here...

        # if we are doing a fast update, just use one of the hyperparameter samples
        # avg_hypers = function_over_hypers if not fast_update else function_over_hypers_single
        avg_hypers = function_over_hypers

        # Compute the acquisition function on the grid
        # self.models.values() returns model instances...... GP models of objective and constraints
        # by using avg_hypers averaging acquisition function values of each hyperparameter
        # dealing state code is in the function_over_hypers function
        # because we use all the grid by x_hat_cands, so code below is not need at all.
        grid_acq = avg_hypers(self.models.values(), acquisition_function,
                                        grid, compute_grad=False, tasks=tasks)

        
        

        # The index and value of the top grid point
        best_acq_ind = np.argmax(grid_acq)
        best_acq_location = grid[best_acq_ind]
        best_grid_acq_value  = np.max(grid_acq)

        has_grads = self.acquisition_function_instance.has_gradients

        if self.options['optimize_acq']:

            if self.options['check_grad']:
                check_grad(lambda x: avg_hypers(self.models.values(), acquisition_function, 
                    x, compute_grad=True), best_acq_location)

            if nlopt_imported:

                # select and specify algorithm
                alg = self.nlopt_method if has_grads else self.nlopt_method_derivative_free

                # specify optimizer
                opt = nlopt.opt(alg, self.num_dims)

                logging.info('Optimizing %s with NLopt, %s' % (self.acquisition_function_name, opt.get_algorithm_name()))
                
                opt.set_lower_bounds(0.0)
                opt.set_upper_bounds(1.0)

                # define the objective function
                def f(x, put_gradient_here):
                    if x.ndim == 1:
                        x = x[None,:]

                    if put_gradient_here.size > 0:
                        a, a_grad = avg_hypers(self.models.values(), acquisition_function, 
                                x, compute_grad=True, tasks=tasks)
                        put_gradient_here[:] = a_grad.flatten()
                    else:
                        a = avg_hypers(self.models.values(), acquisition_function,
                                x, compute_grad=False, tasks=tasks)

                    return float(a)

                # we're finding maximum
                opt.set_max_objective(f)

                # setting ε-optimal condition ε
                opt.set_xtol_abs(self.options['fast_opt_acq_tol'] if fast_update else self.options['opt_acq_tol'])

                # how many grids in the acq_grid_size??
                opt.set_maxeval(self.options['fast_acq_grid_size'] if fast_update else self.options['acq_grid_size'])


                opt_result_points = []
                best_acq_locations = []
                best_acq_values =[]

                # using x_hat_candidates.......not the best in current best 
                for point in x_hat_cands:
                    cur_x_opt = opt.optimize(point.copy())

                    cur_returncode = opt.last_optimize_result()

                    cur_y_opt = f(cur_x_opt, np.array([]))

                    opt_result_points.append(cur_x_opt)

                    if((cur_returncode > 0 or cur_returncode == -4) and cur_y_opt > best_grid_acq_value):
                        best_acq_locations.append(cur_x_opt)
                        best_acq_values.append(cur_y_opt)

                    if(len(best_acq_locations) > 0):
                        best_acq_value = np.max(best_acq_values)
                        best_acq_location = best_acq_locations[np.argmax(best_acq_values)]
                    else:
                        best_acq_value = best_grid_acq_value


                # x_opt = opt.optimize(best_acq_location.copy())

                # returncode = opt.last_optimize_result()
                # # y_opt = opt.last_optimum_value()
                # y_opt = f(x_opt, np.array([]))


                # # overwrite the current best if optimization succeeded
                # # if return code is positive, that means optimization is succesfully done
                # # return code == -4 means roundoff error and that returns useful result....
                # if (returncode > 0 or returncode==-4) and y_opt > best_grid_acq_value:
                #     print_nlopt_returncode(returncode, logging.debug)

                #     best_acq_location = x_opt
                #     best_acq_value = y_opt
                # else:
                #     best_acq_value = best_grid_acq_value

            else: # use bfgs
                # see http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
                logging.info('Optimizing %s with L-BFGS%s' % (self.acquisition_function_name, '' if has_grads else ' (numerically estimating gradients)'))

                if has_grads:
                    def f(x):
                        if x.ndim == 1:
                            x = x[None,:]
                        a, a_grad = avg_hypers(self.models.values(), acquisition_function, 
                                    x, compute_grad=True, tasks=tasks)
                        return (-a.flatten(), -a_grad.flatten())
                else:
                    def f(x):
                        if x.ndim == 1:
                            x = x[None,:]

                        a = avg_hypers(self.models.values(), acquisition_function, 
                                    x, compute_grad=False, tasks=tasks)

                        return -a.flatten()
                
                bounds = [(0,1)]*self.num_dims
                x_opt, y_opt, opt_info = spo.fmin_l_bfgs_b(f, best_acq_location.copy(), 
                    bounds=bounds, disp=0, approx_grad=not has_grads)
                y_opt = -y_opt
                # make sure bounds are respected
                x_opt[x_opt > 1.0] = 1.0
                x_opt[x_opt < 0.0] = 0.0


                if y_opt > best_grid_acq_value:
                    best_acq_location = x_opt
                    best_acq_value = y_opt
                else:
                    best_acq_value = best_grid_acq_value

            logging.debug('Best %s before optimization: %f' % (self.acquisition_function_name, best_grid_acq_value))
            logging.debug('Best %s after  optimization: %f' % (self.acquisition_function_name, best_acq_value))

        else:
            # do not optimize the acqusition function
            logging.debug('Best %s on grid: %f' % (self.acquisition_function_name, best_grid_acq_value))

        return {"location" : best_acq_location, "value" : best_acq_value}
