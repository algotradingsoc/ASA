def optimise_expected_return(Agent, test_cases, verbose=False, **kwargs):
    test_balances = []
    test_params = []
    keys, values = test_cases.keys(), list(test_cases.values())
    check_sublists_same_size(values)
    for current_test_params in zip(*values):
        current_test_dict = dict(zip(keys, current_test_params))
        test_params.append(current_test_dict)
        if verbose:
            print(f'Testing with: {current_test_dict}')
        agent_params = {**current_test_dict, **kwargs}
        agent = Agent(**agent_params)
        agent.run()
        bal = agent.balance
        test_balances.append(bal)
    return list(zip(test_params, test_balances))


def check_sublists_same_size(test_list):
    if type(test_list) is not list:
        raise TypeError(f'Not a list: {type(test_list)}')
    len_first = len(test_list[0])
    assert all(len(i) == len_first for i in test_list), 'Sublists not of same size'

    
if __name__=='__main__':
    from agent_1_simple_macd import SimpleMACDAgent
    test_cases = {'fast_length':[ 5,10,20,20,60],
                  'slow_length':[25,25,40,80,120]}
    outputs = optimise_expected_return(SimpleMACDAgent, test_cases,
                                       backtest='data/backtest_GBPUSD_12_hours.csv')
    print(outputs)
    