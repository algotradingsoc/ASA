# Setup Helper Functions


def setup_agent_0(backtest='data/backtest_GBPUSD_tiny.csv'):
    from Intro_Agents.agent_0_echo import EchoAgent
    agent = EchoAgent(backtest=backtest)
    agent.run()


def setup_agent_1(backtest, verbose):
    from Intro_Agents.agent_1_simple_macd import SimpleMACDAgent
    agent = SimpleMACDAgent(fast_length=120, slow_length=250,
                            backtest=backtest, verbose=verbose)
    agent.run()


def setup_agent_2(backtest, verbose):
    from Intro_Agents.agent_2_simple_risk_managed_macd import SimpleRiskMACDAgent
    agent = SimpleRiskMACDAgent(stop_loss_scaling=2.0,
                                take_profit_scaling=1.5,
                                fast_length=30, slow_length=120,
                                backtest=backtest, verbose=verbose)
    agent.run()


# Test Functions


def test_agent_0(backtest='data/backtest_GBPUSD_tiny.csv'):
    setup_agent_0(backtest=backtest)


def test_agent_1(backtest='data/backtest_GBPUSD_tiny.csv'):
    setup_agent_1(backtest=backtest, verbose=False)
    setup_agent_1(backtest=backtest, verbose=True)


def test_agent_2(backtest='data/backtest_GBPUSD_tiny.csv'):
    setup_agent_2(backtest=backtest, verbose=False)
    setup_agent_2(backtest=backtest, verbose=True)
