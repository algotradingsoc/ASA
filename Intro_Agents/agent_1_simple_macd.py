from pedlar.agent import Agent
from collections import deque
import numpy as np


class SimpleMACDAgent(Agent):
    name = "Simple_MACD"

    def __init__(self,
                 fast_length, slow_length,
                 verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.init_tests(fast_length, slow_length)
        self.fast = deque(maxlen=fast_length) # a object that will drop the oldest item when a new item is added (if it has reached maxlen). https://docs.python.org/3.6/library/collections.html#collections.deque
        self.slow = deque(maxlen=slow_length)
        self.verbose = verbose  # Controls if the agent should print output.
        self.last_mid = None

    def init_tests(self, fast_length, slow_length):
        '''Basic checks to ensure the agent is used correctly.'''
        assert fast_length < slow_length, "Fast length must be \
            less than slow length."

    def on_tick(self, bid, ask, time=None):
        '''Called on every tick update.'''
        mid = (bid + ask) / 2  # we calculate a mid value (a hypothetical value to help us process the tick).
        if self.verbose:
            print(f'Tick: {mid: .05f}, {time}')
        if self.last_mid is None:
            self.last_mid = mid # on the first tick, stores the current mid so the next tick can access it.
            return
        signal = self.get_signal(mid) # get our signal using the current mid as an input.
        self.order_macd(signal) # from the signal decide how to trade.

    def get_signal(self, mid):
        '''Returns the signal from a macd technical indicator.'''
        ret = mid-self.last_mid  # difference between this ticks mid and last ticks mid tells us how much the mid has changed
        self.fast.append(ret) 
        self.slow.append(ret)
        slow_mean = np.mean(self.slow)
        fast_mean = np.mean(self.fast)
        signal = fast_mean - slow_mean
        return signal

    def order_macd(self, signal):
        '''Makes orders based on the signal.'''
        if signal > 0: # positive signal so we want to buy. 
            self.buy()
        elif signal < 0: # negative signal so we want to sell. 
            self.sell()

    def on_order(self, order):
        '''Called on placing a new order.'''
        if self.verbose:
            print('New order:', order)
            print('Orders:', self.orders)  # Agent orders only

    def on_order_close(self, order, profit):
        '''Called on closing an order with some profit.'''
        if self.verbose:
            print('Order closed', order, profit)
            print('Current balance:', self.balance)  # Agent balance only


if __name__ == '__main__':
    backtest = True
    verbose = True
    if backtest:
        # parameters are fed into the agent whilst initialing the agent.
        agent = SimpleMACDAgent(fast_length=120, slow_length=250,
                                verbose=verbose,
                                backtest='../data/backtest_GBPUSD_12_hours.csv')
    else:
        agent = SimpleMACDAgent(fast_length=120, slow_length=250,
                                verbose=verbose,
                                username='joe', password='1234',
                                ticker='tcp://icats.doc.ic.ac.uk:7000',
                                endpoint='http://icats.doc.ic.ac.uk')
    agent.run()
