'''
Code adapted from https://github.com/nuric
An agent that trains a decision tree
as it is running to determine how it should trade.
'''


from Agents.core import Core
from sklearn import tree
import numpy as np

class DecisionTreeAgent(Core):
    name = "Decision_Tree_Agent"

    def __init__(self, fast_length, slow_length,
                 prediction_horizon, max_depth,
                 target_profit, **kwargs):
        super().__init__(**kwargs)
        self.init_tests(fast_length, slow_length,
                        prediction_horizon, max_depth,
                        target_profit)
        self.x_train = []
        self.y_train = []
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.tree = tree.DecisionTreeClassifier(max_depth=max_depth)
        self.max_depth = max_depth
        self.horizon = prediction_horizon
        self.target_profit = target_profit

    def init_tests(self, fast_length, slow_length,
                   prediction_horizon, max_depth,
                   target_profit):
        assert fast_length < slow_length, "Fast length must be less than slow length."
        assert fast_length > 0, "Fast length must be more than zero"
        assert self.rets_length is not None, "Must define a rets length"
        assert self.rets_length >= slow_length, "Rets length must be at least slow length"
        assert prediction_horizon > 0, "Prediction Horizon must be greater than zero."
        assert max_depth > 0, "Must have positive max depth"
        assert target_profit > 0, "Target profit must be greater than zero"

    def core_on_tick(self, bid, ask, time):
        signal = self.get_tree_signal()
        if signal is None:
            return
        self.set_signal(signal)
        self.order(bid, ask)

    def get_tree_signal(self):
        if len(self.rets) < self.rets.maxlen:
            return None
        fast_mean = np.mean(np.array(self.rets)[-self.fast_length:])
        slow_mean = np.mean(self.rets)
        signal = None
        try:
            signal = self.tree.predict([[fast_mean, slow_mean]])[0]
        except Exception as E:
            if self.verbose:
                print(f"Fitting exception: {E}")
            if not self.is_order_open:
                signal = 1
            else:
                signal = 0
        return signal

    def order(self, bid, ask):
        signal = self.signal_value
        if signal == 1:
            self.core_buy(bid, ask)
        elif signal == -1:
            self.core_sell(bid, ask)
        if (self.order_length > self.horizon) and self.is_order_open:
            self.core_close(bid, ask)

    def core_order_open(self, est_order_open_price, order_type, bid, ask):
        fast_avg = np.mean(np.array(self.rets)[-self.fast_length:])
        slow_avg = np.mean(self.rets)
        self.x_train.append([fast_avg, slow_avg])

    def core_on_order_close(self, est_profit, est_order_open_price, order_type):
        y = 0
        if est_profit > self.target_profit:
            if self.order_type == "buy":
                y = 1
            else:
                y = -1
        elif est_profit < self.target_profit:
            if self.order_type == "buy":
                y = -1
            else:
                y = 1
        self.y_train.append(y)
        self.tree.fit(self.x_train, self.y_train)


def main(fast_length=120, slow_length=250,
         prediction_horizon=15, max_depth=2,
         target_profit=0.01, signal_mean_length=10,
         make_orders=True, verbose=True, backtest=None):
    if backtest is None:
        agent = DecisionTreeAgent(fast_length=fast_length,
                                  slow_length=slow_length,
                                  prediction_horizon=prediction_horizon,
                                  max_depth=max_depth,
                                  target_profit=target_profit,
                                  rets_length=slow_length,
                                  signal_mean_length=signal_mean_length,
                                  make_orders=make_orders,
                                  verbose=verbose,
                                  username="joe", password="1234",
                                  ticker="tcp://icats.doc.ic.ac.uk:7000",
                                  endpoint="http://icats.doc.ic.ac.uk")
    else:
        agent = DecisionTreeAgent(fast_length=fast_length,
                                  slow_length=slow_length,
                                  prediction_horizon=prediction_horizon,
                                  max_depth=max_depth,
                                  target_profit=target_profit,
                                  rets_length=slow_length,
                                  signal_mean_length=signal_mean_length,
                                  make_orders=make_orders,
                                  verbose=verbose,
                                  backtest=backtest)
    agent.core_run()
    print(f"Final session balance accurate est: {agent.est_balance[0]:.03f}")
    print(f"Final session balance est: {agent.est_balance[1]:.03f}")
    print("--------------")
