from pedlar.agent import Agent

from risk import RiskMetrics
from core import AgentCore, Buffer

import numpy as np

class TemplateAgent(Agent):
    name = "template agent"
    
    def __init__(self, verbose=False, **kwargs):
        
        self.verbose = verbose
        self.agent_core = AgentCore()
        self.risk = RiskMetrics() 
        super().__init__(**kwargs)
        
        
        
    def on_tick(self, bid, ask):
        if self.verbose:
            print(f"Bid: {bid}, Ask: {ask}")
        self.agent_core.update_bid_ask_mid_spread(bid, ask)
        if self.orders:
            order = self.orders[self._last_order_id]
            self.agent_core.update_order(order)
            self.risk.update_current(self.agent_core.diff)
    
    
    def on_order(self, order):
        pass
    
    
    def on_order_close(self, order, profit):
        self.risk.close_current(profit)
        if self.verbose:
            print(f"Profit: {profit}")
    
    
if __name__=='__main__':
    backtest = False
    if backtest:
        filename="data/1yr_backtest_GBPUSD.csv"
        agent = TemplateAgent(backtest=filename)
    else:
        agent = TemplateAgent(username="algosoc",
                          password="1234",
                          ticker="tcp://icats.doc.ic.ac.uk:7000",
                          endpoint="http://icats.doc.ic.ac.uk")
    agent.run()
    
    if backtest:
        agent.risk.cumulative_return()