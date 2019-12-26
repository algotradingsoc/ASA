from pedlar.agent import Agent
from collections import deque
import numpy as np


class SimpleRiskMACDAgent(Agent):
    '''An improved MACD trading agent with a \
        static stop loss and take profit.'''
    name = "Simple_Risk_MACD"

    def __init__(self,
                 stop_loss_scaling,
                 take_profit_scaling,
                 fast_length, slow_length,
                 verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.init_tests(fast_length, slow_length,
                        stop_loss_scaling, take_profit_scaling)

        self.take_profit_scaling = take_profit_scaling
        self.stop_loss_scaling = stop_loss_scaling
        self.fast = deque(maxlen=fast_length)
        self.slow = deque(maxlen=slow_length)
        self.verbose = verbose

        self.last_spread = None
        self.last_mid = None
        self.last_signal = 0
        self.last_order = -1

    def init_tests(self, fast_length, slow_length,
                   stop_loss_scaling, take_profit_scaling):
        assert fast_length < slow_length, "Fast length must be \
            less than slow length."
        assert stop_loss_scaling > 1.0, "Stop scaling must be \
            large than 1 or else it will instantly close positions."
        assert take_profit_scaling > 0.0, "Take profit must be \
            greater than zero or else it will never make a profit."

    def on_tick(self, bid, ask, time=None):
        '''Called on every tick update.'''
        mid = (bid + ask) / 2
        spread = ask - bid
        if self.last_mid is None:
            self.last_mid = mid
            self.last_spread = spread
            return
        if self.verbose:
            print(f"Tick: {mid: .05f}, {time}")
        signal = self.get_signal(mid)

        is_new_signal = np.sign(signal) != np.sign(self.last_signal)
        if is_new_signal:
            if self.verbose:
                print(f"New signal: {signal}, {self.last_signal}")
            self.last_signal = self.order_macd(signal)
        self.check_take_profit_stop_loss(bid, ask)
        self.last_mid = mid
        self.last_spread = spread

    def get_signal(self, mid):
        '''Returns the signal from a macd technical indicator'''
        ret = mid-self.last_mid
        self.fast.append(ret)
        self.slow.append(ret)
        slow_mean = np.mean(self.slow)
        fast_mean = np.mean(self.fast)
        signal = fast_mean - slow_mean
        return signal

    def order_macd(self, signal):
        '''Makes orders based on the signal'''
        if signal > 0:
            self.buy()
        elif signal < 0:
            self.sell()

    def check_take_profit_stop_loss(self, bid, ask):
        ''' If an order is open, will check if the agent should
            take a profit or stop the loss.
            - Closes any open order if stop loss or take profit.
        '''
        if self.orders:
            o = self.orders[self.last_order]
            diff = bid - o.price if o.type == "buy" else o.price - ask
            if self.verbose:
                print(f"Gross profit: {diff: .05f}")
            if diff > (self.take_profit):
                if self.verbose:
                    print(f"Take profit: \
                        {diff: .05f} > {self.take_profit: .05f}")
                self.close()
            if diff < (self.stop_loss):
                if self.verbose:
                    print(f"Stop loss: \
                        {diff: .05f} < {self.stop_loss: .05f}")
                self.close()

    def on_order(self, order):
        '''Called on placing a new order.'''
        self.last_order = order.id
        self.stop_loss = -self.last_spread * self.stop_loss_scaling
        self.take_profit = self.last_spread * self.take_profit_scaling
        if self.verbose:
            print("New order:", order)
            print("Orders:", self.orders)  # Agent orders only
            print(f"Order detected; \
                take profit: {self.take_profit: .05f}, \
                stop loss: {self.stop_loss: .05f}")

    def on_order_close(self, order, profit):
        '''Called on closing an order with some profit.'''
        if self.verbose:
            print("Order closed", order, profit)
            print("Current balance:", self.balance)  # Agent balance only


if __name__ == "__main__":
    backtest = True
    verbose = True
    if backtest:
        agent = SimpleRiskMACDAgent(stop_loss_scaling=2,
                                    take_profit_scaling=1.5,
                                    fast_length=120, slow_length=250,
                                    verbose=verbose,
                                    backtest='../data/backtest_GBPUSD_12_hours.csv')
    else:
        agent = SimpleRiskMACDAgent(stop_loss_scaling=2,
                                    take_profit_scaling=1.5,
                                    fast_length=120, slow_length=250,
                                    verbose=verbose,
                                    username='joe', password='1234',
                                    ticker='tcp://icats.doc.ic.ac.uk:7000',
                                    endpoint='http://icats.doc.ic.ac.uk')
    agent.run()
