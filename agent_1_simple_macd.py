from pedlar.agent import Agent
from collections import deque
import numpy as np

class SimpleMACDAgent(Agent):
    
    def __init__(self, fast_length, slow_length, 
                 verbose=False, **kwargs):
        super().__init__(**kwargs)
        assert fast_length < slow_length
        self.fast = deque(maxlen=fast_length)
        self.slow = deque(maxlen=slow_length)
        self.verbose = verbose
        
    def on_tick(self, bid, ask, time=None):
        '''Called on every tick update.'''
        if self.verbose:
            print('Tick:', bid, ask, time)
        mid = (bid  + ask) / 2 
        self.fast.append(mid)
        self.slow.append(mid)
        slow_mean = np.mean(self.slow)
        fast_mean = np.mean(self.fast)
        signal = fast_mean - slow_mean
        
        if signal == 0:
            pass
        elif signal > 0:
            self.buy()
        else:
            self.sell()
            
    def on_order(self, order):
        '''Called on placing a new order.'''
        if self.verbose:
            print('New order:', order)
            print('Orders:', self.orders) # Agent orders only

    def on_order_close(self, order, profit):
        '''Called on closing an order with some profit.'''
        if self.verbose:
            print('Order closed', order, profit)
            print('Current balance:', self.balance) # Agent balance only


if __name__ == '__main__':
    backtest = False
    if backtest:
        agent = SimpleMACDAgent(backtest='data/backtest_GBPUSD.csv')
    else:
        agent = SimpleMACDAgent(username='joe', password='1234',
                                ticker='tcp://icats.doc.ic.ac.uk:7000',
                                endpoint='http://icats.doc.ic.ac.uk')
    agent.run()    