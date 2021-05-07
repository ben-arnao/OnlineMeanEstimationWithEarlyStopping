import numpy as np

# This is a small algo i wrote for estimating a fixed value when there is a lot of variance/noise present.
# the algo incorporates a way to define mathematically when our estimator is "stationary" and therefore has arrived
# at the true value, and only fluctuates due to noise.

# min_param_change_perc: the percentage of parameters that need to be considered "stationary" to add to the 
# estimation complete counter

# param_change_delta: the change in estimation results from a new sample to the moving avergae needs to fall below 
# this value for a parameter update to be considered stationary

# param_change_patience: the amount of iterations the estimator needs to remain "stationary" 
# for estimation to be considered complete

# momentum: the momentum of the moving average estimation


class Estimator:

    def __init__(self,
                 num_params,
                 min_param_change_perc=1e-2,
                 param_change_delta=1e-2,
                 param_change_patience=5,
                 momentum=0.9
                 ):

        self.num_params = num_params
        self.momentum = momentum
        self.min_param_change_perc = min_param_change_perc
        self.param_change_delta = param_change_delta
        self.param_change_patience = param_change_patience
        self.param_change_wait = 0

        self.mov_avg = np.zeros(num_params, dtype=np.float32)
        self.hi = np.zeros(num_params, dtype=np.float32)
        self.lo = np.zeros(num_params, dtype=np.float32)

    def estimate(self, new_sample):
        self.mov_avg = self.mov_avg * self.momentum + new_sample * (1 - self.momentum)

        is_new_high = np.where(self.mov_avg > self.hi + self.param_change_delta, True, False)
        is_new_low = np.where(self.mov_avg < self.lo - self.param_change_delta, True, False)

        self.hi = np.where(is_new_high, self.mov_avg, self.hi)
        self.lo = np.where(is_new_low, self.mov_avg, self.lo)

        hi_delta_perc = np.sum(is_new_high) / self.num_params
        lo_delta_perc = np.sum(is_new_low) / self.num_params

        delta_perc = hi_delta_perc + lo_delta_perc

        if delta_perc < self.min_param_change_perc:
            self.param_change_wait += 1
        else:
            self.param_change_wait = 0

        if self.param_change_wait > self.param_change_patience:
            return True
        return False
