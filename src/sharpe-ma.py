""" Sharpe Ratio """
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pedlar.agent import Agent
import time as tm



class SharpeAgent(Agent):
    name = "sharpe"
    def __init__(self, **kwargs):
        self.slow = deque(maxlen=128)
        self.fast = deque(maxlen=32)
        self.last_order = None
        self.price_list=[]
        self.std=0
        self.total_profit_list=[]
        self.total_profit=0
        self.order_in_progress=False
        super().__init__(**kwargs)


    # print order when placing an order
    def on_order(self, order):
        """On order handler."""
        print('----------------------------------------------')
        print("ORDER: ", order)
        self.order_in_progress=True
        self.last_order=order.id
        self.price_list=[]

    # print the profit after the order is closed
    def on_order_close(self, order, profit):
        """On order close handler."""
        print("PROFIT:", profit)
        print('----------------------------------------------')
        self.order_in_progress=False
        self.total_profit+=profit
        self.total_profit_list.append(self.total_profit)



    def on_tick(self, bid, ask, time=None):
        """On tick handler."""
        print("Tick:", bid, ask)
        if not (self.orders):
            #print('no order')
            self.slow.append(bid)
            self.fast.append(bid)
            if np.std(self.fast) == 0 or np.std(self.slow) == 0:
                return
            fast_avg = sum(self.fast)/len(self.fast)
            slow_avg = sum(self.slow)/len(self.slow)
            fast_sharpe = (fast_avg ) / np.std(self.fast)   # use std to measure volatility
            slow_sharpe = (slow_avg ) / np.std(self.slow)
            if fast_avg!=slow_avg:
                print('-------start making trades-------')
                if fast_avg > slow_avg:
                    self.buy()
                else:
                    self.sell()
                return


        # when the sharpe ratio for current value > 1, close the order
        if self.orders:
            o = self.orders[self.last_order]
            print('with order', o.id)
            if o.type=='buy':
                self.price_list.append(bid)
                self.std=np.std(self.price_list)
                sharpe = (bid - o.price )/self.std
            elif o.type == "sell":
                self.price_list.append(ask)
                self.std=np.std(self.price_list)
                sharpe=(o.price - ask )/self.std

            print('sharpe:',sharpe)

            if (sharpe > 1):
                print('-----------------close order')
                print('sharpe: '+str(sharpe))
                self.close()
            return

if __name__ == "__main__":
    backtest=False
    if not backtest:
        agent = SharpeAgent(username="memes", password="memes",
                            ticker="tcp://icats.doc.ic.ac.uk:7000",
                            endpoint="http://icats.doc.ic.ac.uk")
    else:
        agent = SharpeAgent(backtest="backtest_GBPUSD.csv")
  # OR agent = SharpeAgent.from_args() # python3 sharpe.py -b backtest_GBPUSD.csv
    agent.run()
