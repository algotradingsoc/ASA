'''
A simple voting agent that takes the mean of n number of signals.
'''

from pedlar.agent import Agent
from Agents.signal import Signal
from collections import deque
import numpy as np


class SimpleSingalVotingAgent(Agent):
    name = "Mean_singal_voting_agent"

    def __init__(self, agents_list, agents_params, agents_require_pos_func,
                 rets_length, make_order=True, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.init_tests(agents_list, agents_params, agents_require_pos_func)

        self.init_agents_insts(agents_list, agents_params, **kwargs)
        self.agents_req_pos_func = agents_require_pos_func

        self.verbose = verbose
        self.last_mid = None
        self.rets = deque(maxlen=rets_length)
        self.make_order = make_order
        self.signal = Signal(False, None, None)

    def init_agents_insts(self, agents_list, agents_params, **kwargs):
        self.agents_inst = []
        for a, p in zip(agents_list, agents_params):
            combined_dict = {**kwargs, **p}
            self.agents_inst.append(a(**combined_dict))

    def init_tests(self, agents_list, agents_params, agents_require_pos_func):
        num_agents = len(agents_list)
        num_params = len(agents_params)
        num_pos_func_bool = len(agents_require_pos_func)
        assert (num_agents == num_params == num_pos_func_bool), f'{num_agents}!={num_params}!={num_pos_func_bool}'

    def on_tick(self, bid, ask, time=None):
        mid = (bid + ask) / 2
        if self.verbose:
            print(f'Tick: {mid: .05f}, {time}')
        if self.last_mid is None:
            self.last_mid = mid
            return
        ret = np.log(mid) - np.log(self.last_mid)
        self.rets.append(ret)
        if len(self.rets) < self.rets.maxlen:
            return
        self.get_mean_signal(bid, ask, time)
        if self.verbose:
            print(f'Signal: {self.signal.value}')
        self.order()

    def get_mean_signal(self, bid, ask, time):
        rolling_std = np.std(self.rets)
        pos_funcs = []
        for a, req_pf in zip(self.agents_inst, self.agents_req_pos_func):
            a.on_tick(bid, ask, time)
            signal = a.signal.value
            if signal is None:
                continue
            if req_pf:
                signal_vol_norm = signal/rolling_std
                pos_func = np.tanh(signal_vol_norm)
            else:
                pos_func = signal
            pos_funcs.append(pos_func)
        if pos_funcs:
            return
        mean_signal = np.mean(pos_funcs)
        self.signal.set_signal_value(mean_signal)

    def order(self):
        signal = self.signal.value
        if signal > 0:
            self.buy()
        elif signal < 0:
            self.sell()

    # def buy(self):
    #     if self.make_order:
    #         super().buy()
    #     self.signal.open("buy")

    # def sell(self):
    #     if self.make_order:
    #         super().sell()
    #     self.signal.open("sell")


if __name__ == '__main__':
    from agent_1_simple_macd import SimpleMACDAgent
    verbose = False
    agents_list = [SimpleMACDAgent, SimpleMACDAgent, SimpleMACDAgent]
    agents_params = [{'fast_length': 120,
                      'slow_length': 250,
                      'make_order': False,
                      'verbose': verbose},
                     {'fast_length': 75,
                      'slow_length': 150,
                      'make_order': False,
                      'verbose': verbose},
                     {'fast_length': 200,
                      'slow_length': 500,
                      'make_order': False,
                      'verbose': verbose}]
    agents_require_pos_func = [True, True, True]
    backtest = True
    if backtest:
        from util import check_if_in_agents
        check_if_in_agents()
        agent = SimpleSingalVotingAgent(agents_list,  agents_params,
                                        agents_require_pos_func,
                                        rets_length=100,
                                        backtest='../data/backtest_GBPUSD_12_hours.csv')
    else:
        agent = SimpleSingalVotingAgent(agents_list, agents_params,
                                        agents_require_pos_func,
                                        rets_length=100,
                                        username='joe', password='1234',
                                        ticker='tcp://icats.doc.ic.ac.uk:7000',
                                        endpoint='http://icats.doc.ic.ac.uk')
    agent.run()
