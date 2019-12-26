'''
A momentum strategy with some risk measures.
Uses the signal from the simple MACD and implements
a static stop loss and take profit on top.
'''

from Agents.core import Core
from Agents.Risk.stop_loss_take_profit import Stop_Loss_Take_Profit
from Agents.signal import get_macd_signal
import numpy as np

class SimpleRiskMACDAgent(Core):
    name = "Simple_Risk_MACD"

    def __init__(self,
                 fast_length, slow_length, 
                 stop_loss_scaling, take_profit_scaling,
                 **kwargs):
        super().__init__(**kwargs)
        self.init_tests(fast_length, slow_length, self.rets_length)
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.SL_TP = Stop_Loss_Take_Profit(stop_loss_scaling, take_profit_scaling) 
        
    def init_tests(self, fast_length, slow_length, rets_length):
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
        self.SL_TP.check_take_profit_stop_loss(self, bid, ask)

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

    def core_order_open(self, est_order_open_price, order_type, bid, ask):
        spread = ask - bid
        self.SL_TP.set_stop_loss_take_profit(spread)

    def core_on_order_close(self, est_profit, est_order_open_price, order_type):
        self.SL_TP.reset_stop_loss_take_profit()


def main(fast_length=120, slow_length=250,
         stop_loss_scaling=2, take_profit_scaling=1.5,
         make_orders=True, verbose=True, backtest=None):
    if backtest is None:
        agent = SimpleRiskMACDAgent(fast_length=fast_length,
                                    slow_length=slow_length,
                                    stop_loss_scaling=stop_loss_scaling,
                                    take_profit_scaling=take_profit_scaling,
                                    rets_length=slow_length,
                                    make_orders=make_orders,
                                    verbose=verbose,
                                    username='joe', password='1234',
                                    ticker='tcp://icats.doc.ic.ac.uk:7000',
                                    endpoint='http://icats.doc.ic.ac.uk')
    else:
        agent = SimpleRiskMACDAgent(fast_length=fast_length,
                                    slow_length=slow_length,
                                    stop_loss_scaling=stop_loss_scaling,
                                    take_profit_scaling=take_profit_scaling,
                                    rets_length=slow_length,
                                    make_orders=make_orders,
                                    verbose=verbose,
                                    backtest=backtest)
    agent.core_run()
