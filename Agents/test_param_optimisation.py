# Setup

from Agents.param_optimisation_random import random_search_max_expected_return
from Agents.param_optimisation_explicit import explicit_search_max_expected_return, \
                                                print_optimisation_outputs


def expicit_optimise_with_agent_1(backtest,
                                  verbose=False, sort=False):
    from Agents.agent_1_simple_macd import SimpleMACDAgent
    test_cases = {'fast_length': [20, 60],
                  'slow_length': [40, 120],
                  'rets_length': 'slow_length',
                  'signal_mean_length': [1, 5]}
    test_param_balances = explicit_search_max_expected_return(
        SimpleMACDAgent, test_cases,
        backtest=backtest,
        verbose=verbose,
        sort=sort
    )
    if verbose:
        print_optimisation_outputs(test_param_balances)
    return test_param_balances


def random_optimise_with_agent_1(backtest,
                                 n=20, verbose=False, sort=True):
    from Agents.agent_1_simple_macd import SimpleMACDAgent
    search_dict = {'fast_length': [5, 400, int],
                   'slow_length': [20, 1000, int],
                   'rets_length': 'slow_length',
                   'signal_mean_length': [1, 25, int]}
    test_param_balances = random_search_max_expected_return(
        SimpleMACDAgent, search_dict, n,
        backtest, verbose, sort
    )
    if verbose:
        print_optimisation_outputs(test_param_balances)
    return test_param_balances


def expicit_optimise_with_agent_2(backtest,
                                  verbose=False, sort=False):
    from Agents.agent_2_simple_risk_managed_macd import SimpleRiskMACDAgent
    test_cases = {'fast_length': [120]*2,
                  'slow_length': [250]*2,
                  'rets_length': 'slow_length',
                  'signal_mean_length': [1, 5],
                  'stop_loss_scaling': [1.1, 1.5],
                  'take_profit_scaling': [3.5, 5.0]}
    test_param_balances = explicit_search_max_expected_return(
        SimpleRiskMACDAgent, test_cases,
        backtest=backtest,
        verbose=verbose,
        sort=sort
    )
    if verbose:
        print_optimisation_outputs(test_param_balances)
    return test_param_balances


def random_optimise_with_agent_2(backtest,
                                 n=20, verbose=False, sort=True):
    from Agents.agent_2_simple_risk_managed_macd import SimpleRiskMACDAgent
    search_dict = {'fast_length': [5, 400, int],
                   'slow_length': [20, 1000, int],
                   'rets_length': 'slow_length',
                   'signal_mean_length': [1, 25, int],
                   'stop_loss_scaling': [1.1, 6.0, float],
                   'take_profit_scaling': [0.1, 6.0, float]}
    test_param_balances = random_search_max_expected_return(
        SimpleRiskMACDAgent, search_dict, n,
        backtest, verbose, sort
    )
    if verbose:
        print_optimisation_outputs(test_param_balances)
    return test_param_balances


def expicit_optimise_with_agent_3(backtest,
                                  verbose=False, sort=False):
    from Agents.agent_3_ret_bound_risk_macd import RetBoundRiskMACDAgent
    test_cases = {'fast_length': [120] * 2,
                  'slow_length': [250] * 2,
                  'rets_length': 'slow_length',
                  'signal_mean_length': [1, 5],
                  'ret_upper_scaling_factor': [2.5, 3.0],
                  'ret_lower_scaling_factor': [2.5, 3.0]}
    test_param_balances = explicit_search_max_expected_return(
        RetBoundRiskMACDAgent, test_cases,
        backtest=backtest,
        verbose=verbose,
        sort=sort
    )
    if verbose:
        print_optimisation_outputs(test_param_balances)
    return test_param_balances


def random_optimise_with_agent_3(backtest,
                                 n=20, verbose=False, sort=True):
    from Agents.agent_3_ret_bound_risk_macd import RetBoundRiskMACDAgent
    search_dict = {'fast_length': [5, 200, int],
                   'slow_length': [150, 300, int],
                   'rets_length': 'slow_length',
                   'signal_mean_length': [1, 25, int],
                   'ret_upper_scaling_factor': [0.1, 6.0, float],
                   'ret_lower_scaling_factor': [0.1, 6.0, float]}
    test_param_balances = random_search_max_expected_return(
        RetBoundRiskMACDAgent, search_dict, n,
        backtest, verbose, sort
    )
    if verbose:
        print_optimisation_outputs(test_param_balances)
    return test_param_balances


def expicit_optimise_with_agent_4(backtest,
                                  verbose=False, sort=False):
    from Agents.agent_4_decision_tree import DecisionTreeAgent
    test_cases = {'fast_length': [120] * 2,
                  'slow_length': [250] * 2,
                  'rets_length': 'slow_length',
                  'signal_mean_length': [1, 5],
                  'prediction_horizon': [200, 250],
                  'max_depth': [1, 3],
                  'target_profit': [0.1, 0.2]}
    test_param_balances = explicit_search_max_expected_return(
        DecisionTreeAgent, test_cases,
        backtest=backtest,
        verbose=verbose,
        sort=sort
    )
    if verbose:
        print_optimisation_outputs(test_param_balances)
    return test_param_balances


def random_optimise_with_agent_4(backtest,
                                 n, verbose=False, sort=True):
    from Agents.agent_4_decision_tree import DecisionTreeAgent
    search_dict = {'fast_length': [5, 200, int],
                   'slow_length': [150, 300, int],
                   'rets_length': 'slow_length',
                   'signal_mean_length': [1, 25, int],
                   'prediction_horizon': [20, 500, int],
                   'max_depth': [1, 5, int],
                   'target_profit': [0.01, 0.5, float]}
    test_param_balances = random_search_max_expected_return(
        DecisionTreeAgent, search_dict, n,
        backtest, verbose, sort
    )
    if verbose:
        print_optimisation_outputs(test_param_balances)
    return test_param_balances


# Tests


backtest_default = 'data/backtest_GBPUSD_12_hours.csv'
n_default = 5
verbose_default = True
sort_default = True


def test_param_agent_1(backtest=backtest_default, n=n_default,
                       verbose=verbose_default, sort=sort_default):
    expicit_optimise_with_agent_1(backtest, verbose, sort)
    random_optimise_with_agent_1(backtest, n, verbose, sort)


def test_param_agent_2(backtest=backtest_default, n=n_default,
                       verbose=verbose_default, sort=sort_default):
    expicit_optimise_with_agent_2(backtest, verbose, sort)
    random_optimise_with_agent_2(backtest, n, verbose, sort)


def test_param_agent_3(backtest=backtest_default, n=n_default,
                       verbose=verbose_default, sort=sort_default):
    expicit_optimise_with_agent_3(backtest, verbose, sort)
    random_optimise_with_agent_3(backtest, n, verbose, sort)


def test_param_agent_4(backtest=backtest_default, n=n_default,
                       verbose=verbose_default, sort=sort_default):
    expicit_optimise_with_agent_4(backtest, verbose, sort)
    random_optimise_with_agent_4(backtest, n, verbose, sort)
