import numpy as np

class Dynamic_Std_Bounds:
    def __init__(self, upper_bound_scaling, lower_bound_scaling):
        self.init_test(upper_bound_scaling, lower_bound_scaling)
        self.rets_scaling_factor = {'upper' : upper_bound_scaling,
                                    'lower' : lower_bound_scaling}
        self.rets_bounds = {'upper' : None,
                            'lower' : None}

    def init_test(self, upper_bound_scaling, lower_bound_scaling):
        assert upper_bound_scaling > 0.0, "Upper bound scaling factor must be a postive float"
        assert lower_bound_scaling > 0.0, "Lower bound scaling factor must be a postive float"

    def check_dynamic_bounds(self, Agent, bid, ask):
        if Agent.is_order_open:
            current_ret = Agent.rets[-1]
            if current_ret > self.rets_bounds['upper']:
                if Agent.verbose:
                    print(f"Upper ret hit with ret: {current_ret}")
                Agent.core_close(bid, ask)
            elif current_ret < self.rets_bounds['lower']:
                if Agent.verbose:
                    print(f"Lower ret hit with ret: {current_ret}")
                Agent.core_close(bid, ask)
            else:
                if Agent.verbose:
                    print(f"Ret: {current_ret}")


    def set_dynamic_bounds(self, Agent, rets_std=None, rets_mean=None):
        if rets_std is None:
            rets_std = np.std(Agent.rets)
        if rets_mean is None:
            rets_mean = np.mean(Agent.rets)
        self.rets_bounds['upper'] = rets_std * self.rets_scaling_factor['upper'] + rets_mean
        self.rets_bounds['lower'] = -1 * rets_std * self.rets_scaling_factor['lower'] + rets_mean
    
    def reset_dynamic_bounds(self):
        self.rets_bounds['upper'] = None
        self.rets_bounds['lower'] = None