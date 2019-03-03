from pedlar.agent import Agent

from risk import RiskMetrics
from core import AgentCore, Buffer


class MACAgent(Agent):
    name = "Moving Average Crossover Agent"
    
    def __init__(self, period=100, verbose=False, **kwargs):
        self.tick_buffer = Buffer(buffer_length=period)
        self.mean_buffer = Buffer(buffer_length=period)
        self.verbose = verbose
        self.agent_core = AgentCore()
        
        self.risk = RiskMetrics() 
        super().__init__(**kwargs)
        
        
    def on_tick(self, bid, ask):
        
        if self.orders:
            order = self.orders[self._last_order_id]
            self.agent_core.update_order(order)
            self.risk.update_current(self.agent_core.diff)
        
        if self.verbose:
            print(bid, ask)
            
        self.agent_core.update_bid_ask_mid_spread(bid, ask, modify_change=True)
        self.tick_buffer.append(self.agent_core.mid)
        current_mean = self.tick_buffer.get_mean()
        self.mean_buffer.append(current_mean)
        
        if len(self.mean_buffer.buffer) < 5:
            return
        
        mean_change = self.mean_buffer.buffer[-1] - self.mean_buffer.buffer[-2]
        mid = self.agent_core.mid
        prev_mid = self.agent_core.previous
        
        if self.verbose:
            print(mean_change, mid, prev_mid)
        
        if mean_change < 0 and (prev_mid > mid):
            self.sell()
        elif mean_change > 0 and (prev_mid < mid):
            self.buy()
        
            
    def on_order(self, order):
        if self.verbose:
            print(order)
    
    
    def on_order_close(self, order, profit):
        self.risk.close_current(profit)
        if self.verbose:
            print(f"Profit: {profit}")
        
    
if __name__=='__main__':
    backtest = True
    if backtest:
        filename="data/1yr_backtest_GBPUSD.csv"
        agent = MACAgent(backtest=filename)
    else:
        agent = MACAgent(username="algosoc",
                         password="1234",
                         ticker="tcp://icats.doc.ic.ac.uk:7000",
                         endpoint="http://icats.doc.ic.ac.uk")
    agent.run()
    agent.risk.cumulative_return()