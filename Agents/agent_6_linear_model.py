'''
Agent that uses a linear model to make decisions.
'''

from Agents.core import Core

class LinearAgent(Core):
    name = "Linear_Agent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def core_on_tick(self, bid, ask, time=None):
        '''Called on every tick update.'''        
        pass

    def core_on_order_close(self, est_profit, est_order_open_price, order_type):
        pass

    def core_order_open(self, est_order_open_price, order_type):
        pass


def main(rets_length, make_orders, verbose=True, backtest=None):
    if backtest is None:
        agent = LinearAgent(rets_length=rets_length,
                            make_orders=make_orders,
                            verbose=verbose,
                            username='joe', password='1234',
                            ticker='tcp://icats.doc.ic.ac.uk:7000',
                            endpoint='http://icats.doc.ic.ac.uk')
    else:
        agent = LinearAgent(rets_length=rets_length,
                            make_orders=make_orders,
                            verbose=verbose,
                            backtest=backtest)
    agent.run()
