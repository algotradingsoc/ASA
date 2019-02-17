from pedlar.agent import Agent
from risk import RiskMetrics
from core import AgentCore
from rl_ml import DeepQNN
from backtest_funcs import print_backtest_status


import socket

import numpy as np

from collections import deque


HOST = '127.0.0.1'
PORT = 65430

class RLAgent(Agent):
    name = "RL_Agent"
    def __init__(self,
                 file_length=None,
                 verbose=False,   ## prints key info
                 visualise=False, ## visualising with bokeh
                 verbose_ticks=False, ## prints ticks
                 debug=False,     ## prints network actions at each step
                 train=True,      ## trains model, false uses current weights
                 load_model=False,## loads pretrained model
                 **kwargs):
        
        
        
        self.constants = {'name': RLAgent.name,
                          'diff_step': 20,
                          'action_size': 4, ## buy, sell, cancel, do nothing
                          'mid': 100, 'mid_ma': 2000,
                          'memory': 1000, 'order_memory': 1000, 
                          'verbose': verbose, 'visualise': visualise,
                          'verbose_ticks': verbose_ticks, 'debug': debug,
                          'train': train, 'load_model': load_model,
                          'backtest_file_length': file_length}
        

            
#         if self.constants['visualise']:  ### sourt out writing to socket
#             msg = '0.0,0.0,0.0,0.0,0.0'
#             self.send_to_socket(msg)
        
        ## Buffers
        self.mid_buffer = deque(maxlen=self.constants['mid'])
        self.mid_ma_buffer = deque(maxlen=self.constants['mid_ma'])
        self.ma_diff_buffer = self._get_max_ma()
        
        ## Variables
        """ Values change during training """
        self.tick_number, self.bar_number = 0, 0
        self.hold = 100                 ## variable
        self.risk = RiskMetrics()
        self.agent_core = AgentCore()
        
        self.constants['inst_state_size'] = len(self.get_inst_inputs())
        self.constants['ma_diff_buffer_size'] = self.ma_diff_buffer.shape[0]
        
        super().__init__(**kwargs)
        
        self.DQ = DeepQNN(self.constants)
        
        
        
    def on_tick(self, bid, ask):
        """ 
        On tick handler
        Returns: None
        """
        
        self.tick_number += 1
        self.agent_core.update_bid_ask_mid_spread(bid, ask, modify_change=True)
                
        self.mid_buffer.append(self.agent_core.mid) 
        mid_ma = np.mean(np.array(self.mid_buffer))
        self.mid_ma_buffer.append(mid_ma)
        self.update_ma_diff_buffer() 
        
        if self.agent_core.check_hold(self.tick_number):
            return 
        
        if self.orders: 
            order = self.orders[self._last_order_id]
            self.agent_core.update_order(order)
            self.risk.update_current(self.agent_core.diff)
            
#             if self.constants['visualise']:
#                 if self.agent_core.order_length % 5 == 0:
#                     msg = 'NA,NA,NA,{:.3f},0.0'.format(self.agent_core.order_length)
#                     self.send_to_socket(msg)

        if self.constants['verbose']:
            print_backtest_status(self.tick_number,
                                  self.constants['backtest_file_length'], 
                                  freq=100)
            if self.constants['verbose_ticks']:
                self.agent_core.print_status(self.orders)
                
        
        inst = self.get_inst_inputs()
        lstm = self.ma_diff_buffer
        self.DQ.memory = self.DQ.main_loop(self.DQ.memory, inst, lstm, self.orders)
        self.act(self.DQ.variables['action'])
        return
    
    
    
    def on_bar(self, bopen, bhigh, blow, bclose):
        """ On bar handler """
        self.bar_number += 1
        if self.constants['verbose_ticks']:
            print("BAR: ", bopen, bhigh, blow, bclose)
        return
    
    
            
    def on_order(self, order):
        """ On order handler """        
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
        text = '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(self._last_order_id,
                                                           profit, 
                                                           self.balance, 
                                                           self.agent_core.order_length,
                                                           self.DQ.variables['rnd_choice'])
        inst = self.get_inst_inputs()
        lstm = self.ma_diff_buffer
        self.DQ.order_memory = self.DQ.main_loop(self.DQ.order_memory, 
                                                 inst, lstm, self.orders,
                                                 reward=profit, done=True,
                                                 new_action=False)   
        
        
        if self.constants['verbose']:
            print(f'EXIT: {text},{self.DQ.order_epsilon: .5f},{self.DQ.empty_epsilon: .5f}')
            
           
#         if self.constants['visualise']: ## Visualises in bokeh 
#             self.send_to_socket(text)
                
        if self.constants['train']:
            self.DQ.replay(self.DQ.memory, self.DQ.batch_size * 4, 
                           self.DQ.model, decay=False)
            self.DQ.replay(self.DQ.order_memory, self.DQ.batch_size, 
                           self.DQ.model)
                            
        if self._last_order_id % 4 == 0:
            """ Saves weights """
            self.DQ.save(f'models/{RLAgent.name}_weights.h5',
                         self.DQ.model)
                  
        self.agent_core.reset_order()
        self.risk.close_current(profit)
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
    
                
    def get_inst_inputs(self):
        inst_inputs = [[self.agent_core.change['bid']], 
                       [self.agent_core.change['ask']], 
                       [self.agent_core.change['spread']], 
                       [self.agent_core.order_dir], 
                       [self.agent_core.diff],
                       [self.risk.current_trade['max_drawdown']], 
                       [self.risk.current_trade['max_upside']]]  
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
    

    

 
                  
#     def send_to_socket(self, msg):
#         """ 
#         Sends message to bokeh server 
        
#         reward time step - long
#         reward - float
        
#         order time step - long
#         inst val - float
#         cum val - double
#         max drawdown - float
#         max upside - float
        
#         order length count - int 
#         wait length count - int
        
#         rnd order exit - bool (1 = yes, 0 = no)    
#         rnd order entry - bool (1 = yes, 0 = no)
        
#         backtest percent - float
#         """
                  
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((HOST, PORT))
#             s.sendall(msg.encode())
#         return
            
            
            
        
if __name__ == "__main__":   
    backtest = True
    if backtest:
        filename="data/1yr_backtest_GBPUSD.csv"
        from backtest_funcs import get_file_length
        length = get_file_length(filename)
        agent = RLAgent(file_length=length, backtest=filename)
    else:
        agent = RLAgent(username="algosoc",
                        password="1234",
                        ticker="tcp://icats.doc.ic.ac.uk:7000",
                        endpoint="http://icats.doc.ic.ac.uk")
    agent.run()
