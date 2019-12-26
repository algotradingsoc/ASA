# Setup Helper Functions


def setup_agent_0(backtest):
    from Agents.agent_0_echo import EchoAgent
    agent = EchoAgent(backtest=backtest)
    agent.run()


def setup_agent_1(make_orders, verbose, backtest):
    from Agents.agent_1_simple_macd import SimpleMACDAgent
    fast, slow = 120, 250
    agent = SimpleMACDAgent(fast_length=fast, slow_length=slow,
                            rets_length=slow,
                            make_orders=make_orders, verbose=verbose, backtest=backtest)
    agent.run()


def setup_agent_2(make_orders, verbose, backtest):
    from Agents.agent_2_simple_risk_managed_macd import SimpleRiskMACDAgent
    fast, slow = 30, 120
    agent = SimpleRiskMACDAgent(fast_length=fast, slow_length=slow,
                                stop_loss_scaling=2.0,
                                take_profit_scaling=1.5,
                                rets_length=slow,
                                make_orders=make_orders, verbose=verbose, backtest=backtest)
    agent.run()


def setup_agent_3(make_orders, verbose, backtest):
    from Agents.agent_3_ret_bound_risk_macd import RetBoundRiskMACDAgent
    fast, slow = 99, 101
    agent = RetBoundRiskMACDAgent(fast_length=fast, slow_length=slow,
                                  ret_upper_scaling_factor=5,
                                  ret_lower_scaling_factor=3.1,
                                  rets_length=slow+20,
                                  make_orders=make_orders, verbose=verbose, backtest=backtest)
    agent.run()


def setup_agent_4(make_orders, verbose, backtest):
    from Agents.agent_4_decision_tree import DecisionTreeAgent
    fast, slow = 120, 250
    prediction_horizon = 100
    max_depth = 4
    prediction_horizon = 20
    target_profit = 0.1
    agent = DecisionTreeAgent(fast_length=fast, slow_length=slow,
                              prediction_horizon=prediction_horizon, 
                              max_depth=max_depth,
                              target_profit=target_profit,
                              rets_length=slow,
                              make_orders=make_orders, verbose=verbose, backtest=backtest)
    agent.run()


# Test Functions

test_file = 'data/backtest_GBPUSD_1_hour.csv'


def test_agent_0(backtest=test_file):
    setup_agent_0(backtest=backtest)

def agent_test_combination(setup_agent, backtest):
    setup_agent(make_orders=True, verbose=False, backtest=backtest)
    setup_agent(make_orders=True, verbose=True, backtest=backtest)
    setup_agent(make_orders=False, verbose=False, backtest=backtest)
    setup_agent(make_orders=False, verbose=True, backtest=backtest)

def test_agent_1(backtest=test_file):
    agent_test_combination(setup_agent_1, backtest=test_file)
    
def test_agent_2(backtest=test_file):
    agent_test_combination(setup_agent_2, backtest=test_file)

def test_agent_3(backtest=test_file):
    agent_test_combination(setup_agent_3, backtest=test_file)

def test_agent_4(backtest=test_file):
    agent_test_combination(setup_agent_4, backtest=test_file)
