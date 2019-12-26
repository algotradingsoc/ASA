class Stop_Loss_Take_Profit:
    def __init__(self, stop_loss_scaling, take_profit_scaling):
        self.init_test(stop_loss_scaling, take_profit_scaling)
        self.rets_scaling_factor = {'stop_loss' : stop_loss_scaling,
                                        'take_profit' : take_profit_scaling}
        self.rets_bounds = {'stop_loss' : None,
                            'take_profit' : None}

    def init_test(self, stop_loss_scaling, take_profit_scaling):
        assert stop_loss_scaling > 1.0, "Stop scaling must be large than 1 or else it will instantly close positions."
        assert take_profit_scaling > 0.0, "Take profit must be greater than zero or else it will never make a profit."

    def check_take_profit_stop_loss(self, Agent, bid, ask):
        if Agent.is_order_open:
            diff = Agent.get_diff(bid, ask, Agent.order_type)
            if diff > self.rets_bounds['take_profit']:
                if Agent.verbose:
                    print(f"Take profit at: {diff}")
                Agent.core_close(bid, ask)
            elif diff < self.rets_bounds['stop_loss']:
                if Agent.verbose:
                    print(f"Stop loss at: {diff}")
                Agent.core_close(bid, ask)
            else:
                if Agent.verbose:
                    print(f"Diff: {diff}")

    def set_stop_loss_take_profit(self, spread):
        self.rets_bounds['stop_loss'] = -1 * spread * self.rets_scaling_factor['stop_loss']
        self.rets_bounds['take_profit'] = spread * self.rets_scaling_factor['take_profit']

    def reset_stop_loss_take_profit(self):
        self.rets_bounds['stop_loss'] = None
        self.rets_bounds['take_profit'] = None