from pedlar.agent import Agent
from collections import deque
import numpy as np


### what is being implemented
# take profit (make 2* moving std deviation)
# stop loss (make 2* moving std deviation)
###########################


class RiskMACDAgent(Agent):
    """An improved MACD trading agent with risk control measures."""
    
    def __init__(self, **kwargs):
        
        self.rets = deque(maxlen=100)
        self.fast = deque(maxlen=30)
        self.slow = deque(maxlen=60)
        
        self.last_order = None
        self.prev_mid = None
        
        self.tcounter = 0
        self.warmup_ticks = 20
        
        self.early_stop = 0.005
        
        super().__init__(**kwargs)

        
    def on_order(self, order):
        """Called on placing a new order."""
        #print("New order:", order)
        #print("Orders:", self.orders) # Agent orders only
        self.last_order = order.id

        
    def on_order_close(self, order, profit):
        """Called on closing an order with some profit."""
        #print("Order closed", order, profit)
        #print("Current balance:", self.balance) # Agent balance only

        
    def on_tick(self, bid, ask, time=None):
        """Called on every tick update."""
        mid = (bid  + ask) / 2 
        #print(f'Tick: {mid: .05f}, {time}')
        warm = self.warmup(mid)
        if warm:
            self.check_risk(bid, ask)
            self.trade()
        self.tcounter += 1
        
        
    def warmup(self, mid):
        if self.prev_mid is None:
            self.prev_mid = mid
            return
        ret = np.log(mid) - np.log(self.prev_mid)
        self.prev_mid = mid
        self.rets.append(ret)
        return True if self.tcounter > self.warmup_ticks else False
    
    
    def check_risk(self, bid, ask):
        profit = self.get_profit(bid, ask)
        
        #print(f'profit: {profit}')
        
        self.check_early_stop(profit)
        
    
    def get_profit(self, bid, ask):
        if self.orders:
            o = self.orders[self.last_order]
            profit = bid - o.price if o.type == "buy" else o.price - ask
            return profit

            
    def check_early_stop(self, profit):
        if profit is not None:
            if profit > self.early_stop:
                print(f'----------------> take profit: {profit}')
                self.close()
            elif profit < (-2 * self.early_stop):
                print(f'----------------> stop loss: {profit}')
                self.close()

                
    def trade(self):
        ret = self.rets[-1]
        #print('ret:', ret)
        self.fast.append(ret)
        self.slow.append(ret)
        slow_mean = np.mean(self.slow)
        fast_mean = np.mean(self.fast)
        signal = fast_mean - slow_mean
        
        if signal == 0:
            pass
        elif signal > 0:
            #print("buy")
            self.buy()
        else:
            #print("sell")
            self.sell()
        

if __name__ == "__main__":
#     agent = MyAgent(username="joe", password="1234",
#                     ticker="tcp://icats.doc.ic.ac.uk:7000",
#                     endpoint="http://icats.doc.ic.ac.uk")
    agent = MyAgent(backtest="data/backtest_GBPUSD.csv")
    agent.run()