from pedlar.agent import Agent
from risk import RiskMetrics
from core import AgentCore

import random

class RandomAgent(Agent):
    def __init__(self, nothing_prob=0.7, choice_on_tick=False, verbose=False, **kwargs):
        
        self.verbose=verbose
        self.choice_on_tick = choice_on_tick ## If true makes a random choice for every tick, else makes a random choice for every bar.
        self.nothing_prob = nothing_prob
        assert 0.0 <= self.nothing_prob <= 1.0 ## checks probability is a valid input
        
        buy_sell = 1 - self.nothing_prob
        self.buy_prob = buy_sell / 2
        self.sell_prob = buy_sell / 2
        
        self.tick_number, self.bar_number = 0, 0
        
        super().__init__(**kwargs) 
        
        self.risk = RiskMetrics() 
        self.agent_core = AgentCore()
    
    
    def on_tick(self, bid, ask, time=None):
        """On tick handler."""
        self.tick_number += 1
        self.agent_core.update_bid_ask_mid_spread(bid, ask)
        
        if self.orders:
            order = self.orders[self._last_order_id]
            self.agent_core.update_order(order)
            self.risk.update_current(self.agent_core.diff)
        if self.verbose:
            print("Tick:", bid, ask)
        if self.choice_on_tick:
            choice = self.rnd_choice()
            if self.verbose:
                print("Choice:", choice)
    
    
    def on_bar(self, bopen, bhigh, blow, bclose, time=None):
        """On bar handler."""
        self.bar_number += 1
        
        if self.verbose:
            print("Bar:", bopen, bhigh, blow, bclose)
        if not self.choice_on_tick:
            choice = self.rnd_choice()
            if self.verbose:
                print("Choice:", choice)

        
    def rnd_choice(self):
        """ Generates a random choice from the init probs """
        rnd_number = random.uniform(0,1)
        if (self.tick_number == 0) or (self.bar_number) == 0:
            choice = 0
        elif rnd_number <= self.nothing_prob: ## do nothing
            choice = 0
        elif rnd_number <= self.nothing_prob + self.buy_prob: ## buy signal
            choice = 1
            self.buy()
        else: ## (rnd_number <= 1.0) sell signal
            choice = -1
            self.sell()
        return choice
    
    
    def on_order_close(self, order, profit):
        self.risk.close_current(profit)
    
    
if __name__ == "__main__":  
    filename="data/1yr_backtest_GBPUSD.csv"
    agent = RandomAgent(choice_on_tick=False, backtest=filename)
    agent.run()
    results = agent.risk.post_analysis()
    
    for i in results:
        print(f"{i}: {results[i]}")