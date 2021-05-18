import numpy as np

# This is a small algo i wrote for estimating a fixed value when there is a lot of variance/noise present.
# the algo incorporates a way to define mathematically when our estimator is "stationary" and therefore has arrived
# at the true value and only fluctuates due to noise. Because it is "online" we do not need to keep a history 
# of parameters, so this method really would only apply when needing to estimate many parameters such that it is
# infeasible to store a large history of the entire parameter space.

# min_param_change_perc: the percentage of parameters that need to be considered "stationary" to add to the 
# estimation complete counter

# param_change_delta: the change in estimation results from a new sample to the moving avergae needs to fall below 
# this value for a parameter update to be considered stationary

# param_change_patience: the amount of iterations the estimator needs to remain "stationary" 
# for estimation to be considered complete

# momentum: the momentum of the moving average estimation


class Estimator:

    def __init__(self,
                 num_params=1,
                 min_param_change_perc=1e-2,
                 param_change_delta=1e-2,
                 param_change_patience=5,
                 momentum=0.9,
                 bounds_mode='flat'  # or 'scaled'
                 ):
        
        self.bounds_mode = bounds_mode
        self.num_params = num_params
        self.momentum = momentum
        if self.num_params == 1:
            self.min_param_change_perc = 1
        else:
            self.min_param_change_perc = min_param_change_perc
        self.param_change_delta = param_change_delta
        self.param_change_patience = param_change_patience
        self.param_change_wait = 0

        self.mov_avg = np.zeros(num_params, dtype=np.float32)
        self.anchor = np.zeros(num_params, dtype=np.float32)

    def estimate(self, new_sample):
        self.mov_avg = self.mov_avg * self.momentum + new_sample * (1 - self.momentum)
        
        if self.bounds_mode == 'flat':
            is_new_high = np.where(self.mov_avg > self.anchor + self.param_change_delta, True, False)
            is_new_low = np.where(self.mov_avg < self.anchor - self.param_change_delta, True, False)
        elif self.bounds_mode == 'scaled':
            is_new_high = np.where(self.mov_avg > self.anchor * (1 + self.param_change_delta), True, False)
            is_new_low = np.where(self.mov_avg < self.anchor / (1 + self.param_change_delta), True, False)
        else:
            raise Exception('Invalid bounds mode supplied')
        
        # params where average went outside it's bounds
        mom_change = np.logical_or(is_new_high, is_new_low)

        # update anchor
        self.anchor = np.where(mom_change, self.mov_avg, self.anchor)

        # get percent of mean estimations that are non-stationary
        est_chg_perc = np.sum(mom_change) / self.num_params

        if est_chg_perc < self.min_param_change_perc:
            self.param_change_wait += 1
        else:
            self.param_change_wait = 0

        if self.param_change_wait > self.param_change_patience:
            return True
        return False
