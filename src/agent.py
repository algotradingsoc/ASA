import os
import socket
import time
import csv
import random
import numpy as np
from struct import pack
from collections import deque
from pedlar.agent import Agent
from ml import DeepQNN

HOST = '127.0.0.1'
PORT = 65430


class GBPUSD_Agent(Agent):
    name = "Agent-Full-Pass"
    def __init__(self, file_length=None, **kwargs):
        """ Initialises the agent """
        verbose = True             
        ## True - prints core results
        visualise = True
        ## True - visualising with bokeh
        verbose_ticks = False      
        ## True - prints ticks
        debug = False              
        ## True - prints network actions at each step
        write = False              
        ## True - exports results to an output csv
        train = True      
        ## True - trains model, false stops model training
        load_model = True
        ## True - loads pretrained weights into network
        
        self.constants = {'name': GBPUSD_Agent.name,
                          'diff_step': 20,
                          'action_size': 4, ## buy, sell, cancel, do nothing
                          'mid': 100, 'mid_ma': 2000,
                          'memory': 1000, 'order_memory': 1000, 
                          'verbose': verbose, 'visualise': visualise,
                          'verbose_ticks': verbose_ticks, 'debug': debug,
                          'write': write, 'train': train, 'load_model': load_model,
                          'backtest_file_length': file_length}
        
        if self.constants['write']:
            open('data/orders.csv', 'w').close()
            
        if self.constants['visualise']:
            msg = '0.0,0.0,0.0,0.0,0.0'
            self.send_to_socket(msg)
        
        ## Buffers
        self.mid_buffer = deque(maxlen=self.constants['mid'])
        self.mid_ma_buffer = deque(maxlen=self.constants['mid_ma'])
        self.ma_diff_buffer = self._get_max_ma()
        
        ## Variables
        """ Values change during training """
        self.tick_number = 0
        self.hold = 100
        self.balance = 0
        self.order_num = 0
        self.last_order = -1
        self.order_dir = None
        self.order_length = 0
        self.mid = None
        self.bid_diff, self.ask_diff = None, None
        self.spread, self.diff = None, None
        self.last_bid, self.last_ask = None, None
        self.max_drawdown, self.max_upside = None, None
        
        self.constants['inst_state_size'] = len(self.get_inst_inputs())
        self.constants['ma_diff_buffer_size'] = self.ma_diff_buffer.shape[0]
        
        ## Load parent classes
        Agent.__init__(self, **kwargs)
        
        self.DQ = DeepQNN(self.constants)
        
        
        
    def on_tick(self, bid, ask):
        """ 
        On tick handler
        Returns: None
        """
        self.update_backtest_status()
        self.update_bid_ask_mid_spread(bid, ask)
        
        self.order_dir, self.diff = 0, 0 ## Order_dir and order diff reset (If in order then updated)
        if self.last_bid is None:
            self.last_bid, self.last_ask = self.bid, self.ask
            return
        self.bid_diff, self.ask_diff = self.bid-self.last_bid, self.ask-self.last_ask ## Gets bid,ask change since last tick
        self.last_bid, self.last_ask = self.bid, self.ask
        
        self.mid_buffer.append(self.mid) 
        mid_ma = np.mean(np.array(self.mid_buffer))
        self.mid_ma_buffer.append(mid_ma)
        self.update_ma_diff_buffer() ## Updates the moving average difference buffer
        
        if self.hold > 0: 
            self.hold -= 1
            if self.constants['verbose'] or self.constants['verbose_ticks']:
                print("Holding:", self.hold)
            return
        
        if self.orders: 
            ## If in order executed
            self.order_length += 1
            
            if self.constants['visualise']:
                if self.order_length % 5 == 0:
                    msg = 'NA,NA,NA,{:.3f},0.0'.format(self.order_length)
                    self.send_to_socket(msg)
                        
            self.update_diff_and_order_dir()
            self.update_drawdown_upside()
            
        self.print_tick_status()
        
        inst = self.get_inst_inputs()
        lstm = self.ma_diff_buffer
        self.DQ.memory = self.DQ.main_loop(self.DQ.memory, inst, lstm, self.orders)
        self.act(self.DQ.variables['action'])
        return
    
    
    
    def on_bar(self, bopen, bhigh, blow, bclose):
        """ On bar handler """
        self.update_backtest_status()
        if self.constants['verbose_ticks']:
            print("BAR: ", bopen, bhigh, blow, bclose)
        return
    
    
            
    def on_order(self, order):
        """ On order handler """
        self.last_order = order.id
        self.order_num += 1
        self.order_length = 0

        self.order_dir = 1 if order.type == "buy" else -1
        self.max_drawdown, self.max_upside = self.spread * -1, self.spread * -1
        if self.constants['verbose']:
            print(f"ORDER:\t{self.spread * 1000: .3f}\t{order.type}\t{self.DQ.variables['rnd_choice']: }")
        
        inst = self.get_inst_inputs()
        lstm = self.ma_diff_buffer
        self.DQ.order_memory = self.DQ.main_loop(self.DQ.order_memory, 
                                                 inst, lstm, self.orders, 
                                                 new_action=False)   
        return

                  
                  
    def on_order_close(self, order, profit):
        """ On order close handler """
        self.balance += profit
        text = '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(self.order_num,
                                                           profit, 
                                                           self.balance, 
                                                           self.order_length,
                                                           self.DQ.variables['rnd_choice'])
        inst = self.get_inst_inputs()
        lstm = self.ma_diff_buffer
        self.DQ.order_memory = self.DQ.main_loop(self.DQ.order_memory, 
                                                 inst, lstm, self.orders,
                                                 reward=profit, done=True,
                                                 new_action=False)   
        self.order_length = 0
        
        if self.constants['verbose']:
            print(f'EXIT: {text},{self.DQ.order_epsilon: .5f},{self.DQ.empty_epsilon: .5f}')
            
        if self.constants['write']: ## Appends to csv 
            with open('performance/orders.csv', 'a') as f:
                f.write(f'{text}\n')
                
        if self.constants['visualise']: ## Visualises in bokeh 
            self.send_to_socket(text)
                
        if self.constants['train']:
            self.DQ.replay(self.DQ.memory, self.DQ.batch_size * 4, 
                           self.DQ.model, decay=False)
            self.DQ.replay(self.DQ.order_memory, self.DQ.batch_size, 
                           self.DQ.model)
                            
        if self.order_num % 4 == 0:
            """ Saves weights """
            self.DQ.save(f'models/{GBPUSD_Agent.name}_weights.h5',
                         self.DQ.model)
        return
    
                   
                  
    def update_bid_ask_mid_spread(self, bid, ask):
        self.bid, self.ask = bid, ask 
        self.mid = (ask + bid)/2
        self.spread = ask - bid
        return
        
    
    def update_ma_diff_buffer(self):
        mids = np.array(self.mid_ma_buffer) ## Converts deque to np.array
        mids = mids[::-self.constants['diff_step']] ## Gets data point every diff_step
        mids = np.reshape(mids, mids.shape[0]) 
        diff_arr = np.diff(mids)            ## Calculates difference between points
        if diff_arr.shape[0] == 0:          
            ## Catches beginning if self.hold is too small so no data is in diff_arr
            return
        ## Replaces the end values of the array to be fed into the RNN
        self.ma_diff_buffer[-len(diff_arr):] = diff_arr[:]  
        return
    
                  
    def update_diff_and_order_dir(self):
        """ 
        Updates current diff and order_dir (order dir) 
        """
        o = self.orders[self.last_order] #Gets current order 
        if o.type =="buy":
            self.diff = self.bid - o.price
            self.order_dir = 1
        else:
            self.diff = o.price - self.ask
            self.order_dir = -1
        return
    
                  
    def update_drawdown_upside(self):
        if self.diff < self.max_drawdown:
            self.max_drawdown = self.diff
        if self.diff > self.max_upside:
            self.max_upside = self.diff
        return
    
    
    def get_inst_inputs(self):
        inst_inputs = [[self.bid_diff], [self.ask_diff], 
                       [self.spread], [self.order_dir], [self.diff],
                       [self.max_drawdown], [self.max_upside]]
        return inst_inputs
         

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
        """ Returns full moving average buffer - used in setup """
        return np.zeros(self.constants['mid_ma'])[::-self.constants['diff_step']]
    
    def update_backtest_status(self):
        self.tick_number += 1
        if self.constants['backtest_file_length'] is not None:
            if self.tick_number % 100 == 0:
                print('Backtest status: {:.3f} %'.format(100 * self.tick_number 
                                                        / self.constants['backtest_file_length']))
    
    def print_tick_status(self):
        """ Displays the tick status after every tick """
        if self.orders:
            if self.constants['verbose'] and self.constants['verbose_ticks']:
                print("{: .5f} |\t{: .5f}\t{: .5f} |\t{: .5f}\t{: .5f}"
                      .format(self.diff, 
                              self.bid_diff, self.ask_diff, 
                              self.max_drawdown, self.max_upside))
        else:
            if self.constants['verbose'] and self.constants['verbose_ticks']:
                print("{: .5f}\t{: .5f}"
                      .format(self.bid_diff, self.ask_diff))
        return
                  
                  
    def send_to_socket(self, msg):
        """ 
        Sends message to bokeh server 
        
        reward time step - long
        reward - float
        
        order time step - long
        inst val - float
        cum val - double
        max drawdown - float
        max upside - float
        
        order length count - int 
        wait length count - int
        
        rnd order exit - bool (1 = yes, 0 = no)    
        rnd order entry - bool (1 = yes, 0 = no)
        
        backtest percent - float
        """
                  
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(msg.encode())
        return
            
            
            
        
if __name__ == "__main__":   
    backtest = True
    if backtest:
        filename="data/backtest_GBPUSD.csv"
        with open(filename, newline='', encoding='utf-16') as csvfile:
            reader = csv.reader(csvfile)
            length = sum(1 for row in reader)
        agent = GBPUSD_Agent(file_length=length, backtest=filename)
    else:
        agent = GBPUSD_Agent(username="algosoc", 
                             password="1234",                                        
                             ticker="tcp://icats.doc.ic.ac.uk:7000",
                             endpoint="http://icats.doc.ic.ac.uk")
    agent.run()
