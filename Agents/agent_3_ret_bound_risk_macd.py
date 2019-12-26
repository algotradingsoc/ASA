'''
An MACD trading agent with bounded movements of the ret from the mean.
- This agent tries to detect if underlying asset suddenly moves away from
  the mean with a larger than expected move.
'''

from Agents.core import Core
from Agents.Risk.dynamic_std_bounds import Dynamic_Std_Bounds
from Agents.signal import get_macd_signal
import numpy as np

class RetBoundRiskMACDAgent(Core):
    name = "Ret_Bounds_Risk_MACD"

    def __init__(self, fast_length, slow_length,
                 ret_upper_scaling_factor, ret_lower_scaling_factor, **kwargs):
        super().__init__(**kwargs)
        self.init_tests(fast_length, slow_length, 
                        self.rets_length)
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.DSB = Dynamic_Std_Bounds(ret_upper_scaling_factor, ret_lower_scaling_factor)

    def init_tests(self, fast_length, slow_length, rets_length):
        assert fast_length < slow_length, "Fast length must be less than slow length."
        assert fast_length > 0, "Fast length must be at least 1."
        assert self.rets_length is not None, "Must define a rets length"
        assert self.rets_length >= slow_length, "Rets length must be at least slow length"  
        
    def core_on_tick(self, bid, ask, time):
        signal = self.get_macd_signal()
        if signal is None:
            return
        self.set_signal(signal)
        self.order(bid, ask)
        self.DSB.set_dynamic_bounds(self)
        self.DSB.check_dynamic_bounds(self, bid, ask)

    def get_macd_signal(self):
        if len(self.rets) < self.rets.maxlen:
            return None
        return get_macd_signal(np.array(self.rets)[-self.fast_length:], self.rets)

    def order(self, bid, ask):
        signal = self.signal_value
        if signal > 0:
            self.core_buy(bid, ask)
        elif signal < 0:
            self.core_sell(bid, ask)

    def core_on_order_close(self, est_profit, est_order_open_price, order_type):
        self.DSB.reset_dynamic_bounds()


def main(fast_length=120, slow_length=250,
         ret_upper_scaling_factor=1.5,
         ret_lower_scaling_factor=2.0,
         make_orders=True, verbose=True, backtest=None):
    if backtest is None:
        agent = RetBoundRiskMACDAgent(fast_length=fast_length,
                                      slow_length=slow_length,
                                      ret_upper_scaling_factor=ret_upper_scaling_factor,
                                      ret_lower_scaling_factor=ret_lower_scaling_factor,
                                      rets_length=slow_length,
                                      make_orders=make_orders,
                                      verbose=verbose,
                                      username='joe', password='1234',
                                      ticker='tcp://icats.doc.ic.ac.uk:7000',
                                      endpoint='http://icats.doc.ic.ac.uk')

    else:
        agent = RetBoundRiskMACDAgent(fast_length=fast_length,
                                      slow_length=slow_length,
                                      ret_upper_scaling_factor=ret_upper_scaling_factor,
                                      ret_lower_scaling_factor=ret_lower_scaling_factor,
                                      rets_length=slow_length,
                                      make_orders=make_orders,
                                      verbose=verbose,
                                      backtest=backtest)
    agent.core_run()
