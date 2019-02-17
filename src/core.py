from collections import namedtuple

class AgentCore():
    def __init__(self, mv_average=None):
        self.bid = None
        self.ask = None
        self.mid = None
        self.spread = None
        self.change = {'bid':None, 'ask':None, 'mid':None, 'spread':None}
        
        self.order_length = 0
        self.diff = None
        self.order_dir = None
        
        self.hold = None ## need a set and an update
        
        if mv_average != None:
            pass
    
    
    def update_bid_ask_mid_spread(self, bid, ask, modify_change=False):
        mid = (ask + bid)/2
        spread = ask - bid
        if modify_change and (self.bid != None):
            self.update_change(bid, ask, mid, spread)
        self.bid, self.ask = bid, ask 
        self.mid = mid
        self.spread = spread
        return
    
    
    def update_change(self,bid, ask, mid, spread):
        self.change['bid'] = bid - self.bid
        self.change['ask'] = ask - self.ask
        self.change['mid'] = mid - self.mid
        self.change['spread'] = spread - self.spread
    
    
    def update_order(self, order):
        assert order ## ensures an order is open therefore a diff is valid
        self.order_length += 1
        self.diff = self.get_diff(order)
        self.order_dir = self.get_order_dir(order)
        return
    
    
    def reset_order(self):
        self.order_length = 0
        self.diff = None
        self.order_dif = None
    
    
    def get_order_dir(self, order):
        return 1 if order.type == "buy" else -1
        
        
    def get_diff(self, order):
        if order.type =="buy":
            diff = self.bid - order.price
        else:
            diff = order.price - self.ask
        return diff
    

    def print_status(self, orders):
        if orders:
            print("Mid:  {: .5f} | Change: {: .5f} | Spread: {: .5f}\nDiff: {: .5f} | Length: {}"
                  .format(self.mid, self.change['ask'], self.spread,
                          self.diff, self.order_length))
        else:
            print(f"Mid: {self.mid: .5f} | Spread: {self.spread: .5f}")
        
        