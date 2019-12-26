from pedlar.agent import Agent
from collections import deque
import numpy as np

class Core(Agent):

    def __init__(self, rets_length=None, make_orders=True, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.rets_length = rets_length
        if self.rets_length:
            self.rets = deque(maxlen=rets_length)
        self.make_orders = make_orders
        self.verbose = verbose

        self.is_order_open = None
        self.order_type = "close"
        self.order_length = None
        self.est_order_open_price = None
        self.signal_value = None
        self.prev_tick = None

        self.est_balance = [0, 0]
    
    def set_make_orders(self, make_orders):
        if self.verbose:
            print(f"Changing make orders to: {make_orders}")
        assert make_orders in [True, False], "make_orders must be true or false"
        self.make_orders = make_orders
        self.order_type = "close"
        self.is_order_open = False
        self.order_length = None

    def set_signal(self, signal_value):
        self.signal_value = signal_value
    
    def on_tick(self, bid, ask, time):
        '''Called on every tick update.'''
        mid = (bid + ask) / 2
        if self.prev_tick is None:
            self.prev_tick = (bid, ask, time)
            return
        prev_mid = (self.prev_tick[0] + self.prev_tick[1])/2
        log_ret = np.log(mid) - np.log(prev_mid)
        if self.verbose:
            print(f'Tick: {mid: .05f}, {log_ret: .06f}, {time}')
        if self.rets_length:
            self.rets.append(log_ret)
        if self.is_order_open:
            self.order_length += 1
        self.core_on_tick(bid, ask, time)
        self.prev_tick = (bid, ask, time)
        
    def on_order_close(self, order, profit):
        if self.verbose:
            print(f"Order closed: {order}")
            print(f"Profit: {profit}") 
        
    def on_order(self, order):
        if self.verbose:
            print(f"Order open: {order}")

    def core_close(self, bid, ask):
        est_profit = None
        if self.check_is_new_order("close"):
            est_profit, pedlar_est_profit = self.get_est_profit(bid, ask, self.order_type)
            self.est_order_open_price = None
            if self.make_orders:
                self.close()
            self.core_on_order_close(est_profit, self.est_order_open_price, self.order_type)
            self.is_order_open = False
            self.order_type = "close"
            self.order_length = None
            self.est_balance[0] += est_profit
            self.est_balance[1] += pedlar_est_profit
            if self.verbose:
                print(f"Close, profit est.: {est_profit}")
        return est_profit

    def core_buy(self, bid, ask):
        self.core_make_order("buy", bid, ask)

    def core_sell(self, bid, ask):
        self.core_make_order("sell", bid, ask)

    def core_make_order(self, otype, bid, ask):
        assert (otype == "buy") or (otype == "sell"), "otype must be buy or sell"
        open_price = bid if (otype == "buy") else ask
        est_profit = None
        if self.check_is_new_order(otype):
            if self.check_closing_opposite_order(otype):
                self.core_close(bid, ask)
            self.est_order_open_price = open_price
            if self.make_orders:
                if (otype == "buy"):
                    self.buy()
                else:
                    self.sell()
            if self.verbose:
                print(f"{otype} open at: {open_price}")
            self.core_order_open(open_price, otype, bid, ask)
            self.is_order_open = True
            self.order_type = otype
            self.order_length = 0
        return est_profit

    def check_is_new_order(self, new_order_type):
        is_new_order = (self.order_type != new_order_type)
        if is_new_order & self.verbose:
            print(f"New order detected of type: {new_order_type}")
        return is_new_order
    
    def check_closing_opposite_order(self, new_order_type):
        """ Checks if a previous order open is opposite to the new order 
        and therefore closes that order."""
        if (self.order_type == "buy") & (new_order_type == "sell"):
            return True
        elif (self.order_type == "sell") & (new_order_type == "buy"):
            return True
        return False

    def get_diff(self, bid, ask, order_type):
        if order_type == "buy":
            diff = bid - self.est_order_open_price
        elif order_type == "sell": 
            diff = self.est_order_open_price - ask
        else:
            raise TypeError("order_type can only be 'buy' or 'sell'")
        return diff

    def get_est_profit(self, bid, ask, order_type):
        est_profit, pedlar_est_profit = None, None
        leverage = 100
        order_vol = 0.01
        diff = self.get_diff(bid, ask, order_type)
        if order_type == "buy":
            est_profit = diff*leverage*order_vol*1000*(1/bid)
            pedlar_est_profit = round(est_profit, 2)
        elif order_type == "sell":
            est_profit = diff*leverage*order_vol*1000*(1/ask)
            pedlar_est_profit = round(est_profit, 2)
        else:
            raise TypeError("order_type can only be 'buy' or 'sell'")
        return est_profit, pedlar_est_profit

    def core_run(self):
        self.run()
        if self.verbose:
            print(f"Core Session accurate balance: {self.est_balance[0]: .03f}")
            print(f"Core pedlar balance: {self.est_balance[1]: .03f}")
            print("--------------")

    def core_on_tick(self, bid, ask, time):
        pass

    def core_order_open(self, est_order_open_price, order_type, bid, ask):
        pass

    def core_on_order_close(self, est_profit, est_order_open_price, order_type):
        pass
