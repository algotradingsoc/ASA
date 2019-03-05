from pedlar.agent import Agent

from risk import RiskMetrics
from core import AgentCore, Buffer

import numpy as np

class MACAgent(Agent):
    name = "Moving Average Crossover"
    
    def __init__(self, ma_1_length=12, ma_2_length=64, verbose=False, **kwargs):
        self.slow_period = max(ma_1_length, ma_2_length)
        self.fast_period = min(ma_1_length, ma_2_length)
        self.fast_mean = None
        self.slow_mean = None
        self.prev_slow_less_fast = None
        
        self.tick_buffer = Buffer(buffer_length=self.slow_period)
        
        self.verbose = verbose
        
        self.agent_core = AgentCore()
        self.risk = RiskMetrics() 
        super().__init__(**kwargs)
        
        
    def on_tick(self, bid, ask, time=None):
        self.agent_core.update_bid_ask_mid_spread(bid, ask, modify_change=True)
        if self.orders:
            order = self.orders[self._last_order_id]
            self.agent_core.update_order(order)
            self.risk.update_current(self.agent_core.diff)
        
        if self.verbose:
            print(f"Bid: {bid: .5f}, Ask: {ask: .5f}")
        
        self.tick_buffer.append(self.agent_core.mid)
        
        self.slow_mean = self.tick_buffer.get_mean()
        self.fast_mean = np.mean(self.tick_buffer.get_array()[-self.fast_period:])
        
        if self.prev_slow_less_fast is None:
            self.update_prev_slow_less_fast()
            return 
        
        if (self.prev_slow_less_fast) and (self.fast_mean > self.slow_mean):
            self.buy()
        elif (not self.prev_slow_less_fast) and (self.fast_mean < self.slow_mean):
            self.sell()
        self.update_prev_slow_less_fast()
        
        
    def update_prev_slow_less_fast(self):
        self.prev_slow_less_fast = self.slow_mean < self.fast_mean 
        
    
    def on_bar(self, bopen, bhigh, blow, bclose, time=None):
        pass
    
        
    def on_order(self, order, time=None):
        if self.verbose:
            print("ORDER")
            print(f"Order: {order}")
    
    
    def on_order_close(self, order, profit, time=None):
        self.risk.close_current(profit)
        if self.verbose:
            print(f"Profit: {profit}")
        
    
if __name__=='__main__':
    backtest = True
    if backtest:
        filename="data/1yr_backtest_GBPUSD.csv"
        agent = MACAgent(backtest=filename, verbose=False)
    else:
        agent = MACAgent(verbose=True,
                         username="algosoc",
                          password="1234",
                          ticker="tcp://icats.doc.ic.ac.uk:7000",
                          endpoint="http://icats.doc.ic.ac.uk")
    agent.run()
    
    if backtest:
        agent.risk.cumulative_return()