from pedlar.agent import Agent

import numpy as np

class TemplateAgent(Agent):
    name = "template agent"
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
    def on_tick(self, bid, ask):
        print(bid, ask)
    
    def on_order(self, order):
        pass
    
    def on_order_close(self, order, profit):
        pass
    
    
    
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