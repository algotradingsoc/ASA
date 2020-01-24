'''
Agent that uses a linear model to make decisions.
'''

from Agents.core import Core
from collections import deque
from sklearn.linear_model import LinearRegression


class LinearAgent(Core):
    name = "Linear_Agent_MSE"

    def __init__(self, train_data_length, predict_ahead, **kwargs):
        super().__init__(**kwargs)
        assert self.rets_length > (1 + predict_ahead), f"Rets length ({self.rets_length}) must be greater than 1+{predict_ahead}."
        assert predict_ahead >= 0, "Must be positive predict ahead."
        self.predict_ahead = int(predict_ahead)
        self.train_data = deque(maxlen=train_data_length)
        self.test_data = deque(maxlen=train_data_length)

        self.warmup = int(train_data_length / 2)
        self.lin_model = LinearRegression()
        self.regressor = None

    def core_on_tick(self, bid, ask, time=None):
        '''Called on every tick update.'''
        if len(self.rets) < self.rets.maxlen:
            return
        train = list(self.rets)[:-(1 + self.predict_ahead)]
        target = self.rets[-1]

        self.train_data.append(train)
        self.test_data.append(target)

        if len(self.test_data) > self.warmup:
            self.regressor = self.lin_model.fit(list(self.train_data),
                                                list(self.test_data))
            pred_y = self.regressor.predict([list(self.rets)[(1 + self.predict_ahead):]])
            self.set_signal(pred_y)
            self.order(bid, ask)

    def order(self, bid, ask):
        signal = self.signal_value
        if signal > 0:
            self.core_buy(bid, ask)
        elif signal < 0:
            self.core_sell(bid, ask)

    def core_order_open(self, est_order_open_price, order_type, bid, ask):
        pass

    def core_on_order_close(self, est_profit, est_order_open_price, order_type):
        pass


def main(train_data_length=20, predict_ahead=10,
         rets_length=30, signal_mean_length=5, make_orders=True,
         verbose=True, backtest=None):
    if backtest is None:
        agent = LinearAgent(train_data_length, predict_ahead,
                            rets_length=rets_length,
                            signal_mean_length=signal_mean_length,
                            make_orders=make_orders,
                            verbose=verbose,
                            username='joe', password='1234',
                            ticker='tcp://icats.doc.ic.ac.uk:7000',
                            endpoint='http://icats.doc.ic.ac.uk')
    else:
        agent = LinearAgent(train_data_length, predict_ahead,
                            rets_length=rets_length,
                            signal_mean_length=signal_mean_length,
                            make_orders=make_orders,
                            verbose=verbose,
                            backtest=backtest)
    agent.run()
    print(f"Final session balance accurate est: {agent.est_balance[0]:.03f}")
    print(f"Final session balance est: {agent.est_balance[1]:.03f}")
    print("--------------")
