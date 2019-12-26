'''
Determine the expected return from using a random search
from an upper and lower bound for each parameter for an agent.
'''

import random


def random_search_max_expected_return(Agent,
                                      search_dictionary, n_steps,
                                      backtest,
                                      verbose=False, sort=True, core=True):
    keys, values = search_dictionary.keys(), list(search_dictionary.values())
    failed_runs = 0
    test_param_balances = []
    for n in range(n_steps):
        agent_inputs = {}
        for idx, k in enumerate(keys):
            if values[idx][2] == int:
                k_value = random.randrange(values[idx][0], values[idx][1])
            elif values[idx][2] == float:
                k_value = random.uniform(values[idx][0], values[idx][1])
            elif type(values[idx]) == str:
                assert values[idx] in agent_inputs, f"{values[idx]} must be agent_inputs"
                agent_inputs[k] = agent_inputs[values[idx]]
            else:
                raise NotImplementedError()
            agent_inputs[k] = k_value
        if verbose:
            print(f"{n}) Using {agent_inputs}")
        try:
            agent = Agent(backtest=backtest, verbose=False, **agent_inputs)
            if not core:
                agent.run()
                test_param_balances.append([agent_inputs, agent.balance])
            else:
                agent.core_run()
                test_param_balances.append([agent_inputs, agent.balance, 
                                            agent.est_balance[0], agent.est_balance[1]])
        except Exception as E:
            failed_runs += 1
            print(f"Run failed: {E}, fail number: {failed_runs}")
    if sort:
        test_param_balances = list(reversed(sorted(test_param_balances,
                                                   key=lambda i: i[1])))
    if verbose:
        for i in test_param_balances:
            print(f"{i[1]: .3f} | {i[0]}")
        print(f"Failed runs: {failed_runs}/{n_steps}")
    return test_param_balances


"""
Example Usage
from Agents.agent_4_decision_tree import DecisionTreeAgent
search_dict = {'horizon': [2, 30, int],
                'max_depth': [1, 4, int],
                'fast_length': [10, 149, int],
                'slow_length': [20, 500, int],
                'rets_length': 'slow_length'}
verbose = True
backtest = '../data/backtest_GBPUSD_12_hours.csv'
test_param_balances = random_search_max_expected_return(DecisionTreeAgent,
                                                        search_dict,
                                                        200,
                                                        backtest=backtest,
                                                        verbose=verbose)
local_optimal_params = test_param_balances[0]
agent = DecisionTreeAgent(verbose=verbose,
                            username='Joe', password='1234',
                            ticker='tcp://icats.doc.ic.ac.uk:7000',
                            endpoint='http://icats.doc.ic.ac.uk',
                            **local_optimal_params[0])
"""
