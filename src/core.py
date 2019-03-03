from collections import deque
import numpy as np

class AgentCore():
    def __init__(self, hold=10):
        self.bid = None
        self.ask = None
        self.mid = None
        self.spread = None
        self.change = {'bid':None, 'ask':None, 'mid':None, 'spread':None}
        self.prev_mid = None
        
        self.order_length = 0
        self.diff = None
        self.order_dir = None
        
        self.hold = hold
    
    
    def check_hold(self, tick):
        if tick < self.hold:
            return True ## (In hold period)
        else:
            return False
    
    
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
        self.prev_mid = self.mid
    
    
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
        

        
class Buffer():
    def __init__(self, buffer_length=100, step=1, rnn_input=False):
        self.buffer_length = buffer_length
        self.step = step
        self.rnn_input = rnn_input
        self.buffer = deque(maxlen=self.buffer_length)
        
        if self.rnn_input:
            self.rnn_input_template = self._get_rnn_in_template()
            pass
    
    
    def append(self, val):
        self.buffer.append(val)
        
        
    def get_array(self):
        """ Returns np.array from the deque """
        return np.array(self.buffer)
    
    
    def get_mean(self):
        arr = self.get_array()
        return np.mean(arr)
    
    
    def get_vals_at_steps_reversed(self):
        """ Returns array with data at every step.
        Starts from end (most recent deque input) """
        arr = self.get_array()
        return arr[::-self.step]
    
    
    def get_diff_array(self, arr=None):
        """ Returns array of differences between steps """
        if arr is None:
            arr = self.get_array()
        return np.diff(arr)
    
    
    def _copy_into_template(self, arr):
        """ From template, copies values in for rnn input """
        copy = self.rnn_input_template
        copy[-len(arr):] = arr[:]
        return copy
    
    
    def get_rnn_input(self):
        assert self.rnn_input ## Buffer not initialised with rnn_input
        arr = self.get_vals_at_steps_reversed()
        diff_arr = self.get_diff_array(arr)
        if diff_arr.shape[0] is 0:
            return self.rnn_input_template
        ## Catches beginning if not enough input is given, possibly too short a hold period
        return self._copy_into_template(diff_arr)
        
    
    def _get_rnn_in_template(self):
        return np.zeros(self.buffer_length)[::-self.step]
    
   