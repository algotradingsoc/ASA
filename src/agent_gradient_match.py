from pedlar.agent import Agent

from risk import RiskMetrics
from core import AgentCore, Buffer


class AIGMAgent(Agent):
    name = "Average and Inst Gradient Match"
    
    def __init__(self, period=100, mean_period=20, mean_compare=3, verbose=False, **kwargs):
        self.tick_buffer = Buffer(buffer_length=period)
        self.mean_buffer = Buffer(buffer_length=mean_period)
        self.mean_compare = mean_compare
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
            print(f"Bid: {bid}, Ask: {ask}")
            
        self.agent_core.update_bid_ask_mid_spread(bid, ask, modify_change=True)
        self.tick_buffer.append(self.agent_core.mid)
        current_mean = self.tick_buffer.get_mean()
        self.mean_buffer.append(current_mean)
        
        if len(self.mean_buffer.buffer) < self.mean_buffer.buffer.maxlen:
            return
                
        mean_diffs = self.mean_buffer.get_diff_array()
        mid = self.agent_core.mid
        prev_mid = self.agent_core.previous
        
        if self.verbose:
            print(f"{mean_diffs[-1]}, {mid}, {prev_mid}")
        
        if mid < prev_mid:
            for i in range(self.mean_compare):
                if not (mean_diffs[-(i+1)] < 0):
                    break
            else:
                self.sell()
        elif mid > prev_mid:
            for i in range(self.mean_compare):
                if not (mean_diffs[-(i+1)] > 0):
                    break
            else:
                self.buy()
            
        
    def on_order(self, order):
        if self.verbose:
            print(f"Order: {order}")
    
    
    def on_order_close(self, order, profit):
        self.risk.close_current(profit)
        if self.verbose:
            print(f"Profit: {profit}")
        
    
if __name__=='__main__':
    backtest = True
    if backtest:
        filename="data/1yr_backtest_GBPUSD.csv"
        agent = AIGMAgent(backtest=filename)
    else:
        agent = AIGMAgent(username="algosoc",
                          password="1234",
                          ticker="tcp://icats.doc.ic.ac.uk:7000",
                          endpoint="http://icats.doc.ic.ac.uk")
    agent.run()
    
    if backtest:
        agent.risk.cumulative_return()