'''
 Simple momentum strategy looking at the
 cross over of two moving averages of the returns.
'''

from Agents.core import Core
from Agents.signal import get_macd_signal
import numpy as np


class SimpleMACDAgent(Core):
    name = "Simple_MACD"

    def __init__(self,
                 fast_length, slow_length,
                 **kwargs):
        super().__init__(**kwargs)
        self.init_tests(fast_length, slow_length)
        self.fast_length = fast_length
        self.slow_length = slow_length

    def init_tests(self, fast_length, slow_length):
        assert fast_length < slow_length, "Fast length must be less than slow length."
        assert fast_length > 0, "Fast length must be more than zero"
        assert self.rets_length is not None, "Must define a rets length"
        assert self.rets_length >= slow_length, "Rets length must be at least slow length"

    def core_on_tick(self, bid, ask, time):
        signal = self.get_macd_signal()
        if signal is None:
            return
        self.set_signal(signal)
        self.order(bid, ask)

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


def main(fast_length=120, slow_length=250, signal_mean_length=10,
         make_orders=True, verbose=True, backtest=None):
    if backtest is None:
        agent = SimpleMACDAgent(fast_length=fast_length,
                                slow_length=slow_length,
                                rets_length=slow_length,
                                signal_mean_length=signal_mean_length,
                                make_orders=make_orders,
                                verbose=verbose,
                                username='joe', password='1234',
                                ticker='tcp://icats.doc.ic.ac.uk:7000',
                                endpoint='http://icats.doc.ic.ac.uk')
    else:
        agent = SimpleMACDAgent(fast_length=fast_length,
                                slow_length=slow_length,
                                rets_length=slow_length,
                                signal_mean_length=signal_mean_length,
                                make_orders=make_orders,
                                verbose=verbose,
                                backtest=backtest)
    agent.core_run()
    print(f"Final session balance accurate est: {agent.est_balance[0]:.03f}")
    print(f"Final session balance est: {agent.est_balance[1]:.03f}")
    print("--------------")
