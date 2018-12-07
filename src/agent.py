import os
import socket
import time
import csv
import random
import numpy as np
from collections import deque
from pedlar.agent import Agent
from ml import DeepQNN

HOST = '127.0.0.1'
PORT = 65430


class GBPUSD_Agent(Agent, DeepQNN):
    name = "GBPUSD-RL-Agent"
    def __init__(self, **kwargs):
        """ Initialises the agent """
        self.verbose = True             ## True for printing core results
        self.visualise = True           ## True for visualising with bokeh
        self.verbose_ticks = False      ## True for printing all results
        self.debug = False              ## True for debugging
        self.write = False              ## True for exporting results to an output csv
        self.train = True               ## True for training the model - false stops the model from training on inputs it recieves
        
        if self.write:
            open('data/orders.csv', 'w').close()
        
        ## Constants
        self.constants = {'diff_step': 20,
                          'mid': 100, 'mid_ma': 2000,
                          'memory': 1000, 'order_memory': 1000}
        
        ## Buffers
        self.mid_buffer = deque(maxlen=self.constants['mid'])
        self.mid_ma_buffer = deque(maxlen=self.constants['mid_ma'])
        self.ma_diff_buffer = self._get_max_ma()
        
        ## Variables
        """ Values change during training """
        self.hold = 100
        self.balance = 0
        self.order_num = 0
        self.last_order = -1
        self.order_dir = None
        self.rnd_order = 0
        self.order_length = 0
        self.mid = None
        self.bid_diff, self.ask_diff = None, None
        self.spread, self.diff = None, None
        self.last_bid, self.last_ask = None, None
        self.max_drawdown, self.max_upside = None, None
        self.state, self.next_state = None, None
        self.inst_state_size = self.get_state()[0].shape[1]  
        
        ## Load parent classes
        Agent.__init__(self, **kwargs)
        DeepQNN.__init__(self, self.train,
                         GBPUSD_Agent.name, 
                         self.inst_state_size)
        
        
    def on_tick(self, bid, ask):
        """On tick handler."""
        self.bid, self.ask, self.spread = bid, ask, ask-bid ## Set bid, ask, spread
        self.mid = (ask - bid)/2
        
        self.order_dir, self.diff = 0, 0 ## Order_dir and order diff reset (If in order then updated)
        
        if self.last_bid is None:
            self.last_bid, self.last_ask = self.bid, self.ask
            return
        self.bid_diff, self.ask_diff = self.bid-self.last_bid, self.ask-self.last_ask ## Gets bid and ask change since last tick
        self.last_bid, self.last_ask = self.bid, self.ask
        
        self.mid_buffer.append(self.mid) 
        self.mid_ma = np.mean(np.array(self.mid_buffer))
        self.mid_ma_buffer.append(self.mid_ma)
        
        self.set_ma_diff_buffer() ## Updates the moving average difference buffer
        
        if self.hold > 0: 
            self.hold -= 1
            if self.verbose or self.verbose_ticks:
                print("Holding:", self.hold)
            return
        
        if self.orders: 
            ## If in order executed
            self.order_length += 1
            if self.order_length % 5 == 0:
                
                if self.visualise:
                    msg = 'NA,NA,NA,{:.3f}'.format(self.order_length)
                    self.send_to_socket(msg)
                        
            o = self.orders[self.last_order] #Gets current order object
            if o.type =="buy":
                self.diff = self.bid - o.price
                self.order_dir = 1
            else:
                self.diff = o.price - self.ask
                self.order_dir = -1
            
            if self.diff < self.max_drawdown:
                self.max_drawdown = self.diff
            if self.diff > self.max_upside:
                self.max_upside = self.diff
                
            if self.verbose and self.verbose_ticks:
                print("{: .5f} |\t{: .5f}\t{: .5f} |\t{: .5f}\t{: .5f}"
                      .format(self.diff, 
                              self.bid_diff, self.ask_diff, 
                              self.max_drawdown, self.max_upside))
        else:
            if self.verbose and self.verbose_ticks:
                print("{: .5f}\t{: .5f}"
                      .format(self.bid_diff, self.ask_diff))
        self.rl_loop()
        
        
        
    def on_bar(self, bopen, bhigh, blow, bclose):
        """On bar handler """
        if self.verbose_ticks:
            print("BAR: ", bopen, bhigh, blow, bclose)
            
            
            
    def on_order(self, order):
        """On order handler."""
        self.last_order = order.id
        self.order_num += 1
        self.order_length = 0

        self.order_dir = 1 if order.type == "buy" else -1
        self.max_drawdown, self.max_upside = self.spread * -1, self.spread * -1
        if self.verbose:
            print(f"ORDER:\t{self.spread * 1000: .3f}\t{order.type}\t{self.rnd_order: }")
        self.order_memory = self.remember(self.order_memory,
                                          self.state, self.action, 
                                          self.reward, self.next_state, 
                                          self.done)

            
    def on_order_close(self, order, profit):
        """On order close handler."""
        self.reward, self.done = profit, True
        self.balance += profit
        text = '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(self.order_num,
                                                           profit, 
                                                           self.balance, 
                                                           self.order_length,
                                                           self.rnd_order)
        self.order_memory = self.remember(self.order_memory,
                                          self.state, self.action, 
                                          self.reward, self.next_state, 
                                          self.done)
        self.order_length = 0
        
        if self.verbose:
            print(f'{text},{self.order_epsilon: .5f},{self.empty_epsilon: .5f}')
            
        if self.write:
            """ Appends to csv """
            with open('performance/orders.csv', 'a') as f:
                f.write(f'{text}\n')
                
        if self.visualise:
            """ Visualises in bokeh """
            self.send_to_socket(text)
                
        if self.train:
            self.replay(self.memory, self.batch_size * 4, 
                        self.model)
            self.replay(self.order_memory, self.batch_size, 
                        self.model, decay=False)
            
                
        if self.order_num % 10 == 0:
            """ Saves weights """
            self.save(f'models/{GBPUSD_Agent.name}_weights.h5', self.model)
    
    
    def rl_loop(self):
        ## RL Logic
        self.reward, self.done = 0, False 
        self.next_state = self.get_state()
        if self.state is None:
            self.state = self.next_state
            return
        self.action = self.get_action(self.state)
        self.act(self.action)
        self.memory.append((self.state, self.action, 
                            self.reward, self.next_state, self.done))
        self.state = self.next_state
            
    
    def get_state(self):
        """ 
        Returns list of MLP inputs and RNN inputs
        MLP: 
         - self.bid_diff       : Change in Bid from last previous
         - self.ask_diff       : Change in Ask from last previous
         - self.spread         : Difference between ask and diff
         - self.order_dir      : Order type (0 = no order, 1 = buy order, 
                                             -1 = sell order)
         - self.max_drawdown   : Difference since the lowest 
                                 point in current order
         - self.max_upside     : Difference since the highest point 
                                 in current order
        RNN:
         - self.ma_diff_buffer : Array of difference in price at 
                                 intervals of diff_step 
        """
        inst_inputs = [[self.bid_diff], [self.ask_diff], 
                       [self.spread], [self.order_dir], [self.diff],
                       [self.max_drawdown], [self.max_upside]]
        mid_buffer = np.array([list(self.ma_diff_buffer)])
        inst = np.reshape(inst_inputs, [1, len(inst_inputs)])
        mid = np.reshape(mid_buffer, (1, mid_buffer.shape[1], 1))
        return [inst, mid]
    
    
    def get_action(self, state):
        """ 
        Determines the action to take, returns action ID.
         - Depends on the state and the epsilon value
         - Makes random choice with probability of epsilon.
           - self.order_epsilon : probability when the model 
                                  is currently in an order
           - self.empty_epsilon : probability when the model 
                                  is not in an order
        """
        if self.orders:
            if np.random.rand() <= self.order_epsilon:
                return self.make_random_choice()
        else:
            if np.random.rand() <= self.empty_epsilon:
                return self.make_random_choice()
        ## If not random choice tries to predict best action
        act_values = self.model.predict([state[0], 
                                         state[1]])
        if self.debug:
            print(np.argmax(act_values[0]), "+")
        self.rnd_order = 0 ## Random choice was not made
        return np.argmax(act_values[0])
    
    
    def make_random_choice(self):
        rnd_choice = random.randrange(self.action_size)
        if self.debug:
            print(rnd_choice, "-")
        self.rnd_order = 1  ## Random choice was made
        return rnd_choice
    
    
    def act(self, action):
        """ 
        Performs action:
         - 1 : buys 
         - 2 : sells 
         - 3 : closes
         - 0 : nothing
        """
        if action == 1:
            self.buy()
        elif action == 2:
            self.sell()
        elif action == 3:
            if self.orders:
                self.close()
        else:
            pass
        return
        
    
    def _get_max_ma(self):
        """ Returns  """
        return np.zeros(self.constants['mid_ma'])[::-self.constants['diff_step']]
    
    
    def set_ma_diff_buffer(self):
        mids = np.array(self.mid_ma_buffer) ## Converts deque to np.array
        mids = mids[::-self.constants['diff_step']] ## Gets data point every diff_step
        mids = np.reshape(mids, mids.shape[0]) 
        diff_arr = np.diff(mids)            ## Calculates difference between points
        if diff_arr.shape[0] == 0:          
            ## Catches beginning if self.hold is too small so no data is in diff_arr
            return
        ## Replaces the end values of the array to be fed into the RNN
        self.ma_diff_buffer[-len(diff_arr):] = diff_arr[:]  

        
        
    def send_to_socket(self, msg):
        """ Sends message to bokeh server """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(msg.encode())
            
            
            
        
if __name__ == "__main__":   
    backtest = True
    if backtest:
        agent = GBPUSD_Agent(backtest="data/backtest_GBPUSD.csv")
    else:
        agent = GBPUSD_Agent(username="algosoc", 
                             password="1234",                                        
                             ticker="tcp://icats.doc.ic.ac.uk:7000",
                             endpoint="http://icats.doc.ic.ac.uk")
    agent.run()
