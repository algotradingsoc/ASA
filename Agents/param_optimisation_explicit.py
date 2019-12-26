'''
Determine the expected return from using an explicitly
defined set of parameters for an agent.
'''


def explicit_search_max_expected_return(Agent, test_cases,
                                        verbose=False, sort=False, core=True,
                                        **kwargs):
    test_param_balances = []
    test_params = []
    keys, values = test_cases.keys(), list(test_cases.values())
    list_length = check_sublists_same_size(values)
    for test_idx in range(list_length):
        current_test_dict = {}
        for idx, key in enumerate(keys):
            if type(values[idx]) is str:
                current_test_dict[key] = test_cases[values[idx]][test_idx]
            else:
                current_test_dict[key] = values[idx][test_idx]
        test_params.append(current_test_dict)
        if verbose:
            print(f'Testing with: {current_test_dict}')
        agent_params = {**current_test_dict, **kwargs}
        agent = Agent(**agent_params)
        if not core:
            agent.run()
            test_param_balances.append([agent_params, agent.balance])
        else:
            agent.core_run()
            test_param_balances.append([agent_params, agent.balance, 
                                        agent.est_balance[0], agent.est_balance[1]])
    if sort:
        test_param_balances = reversed(sorted(test_param_balances,
                                              key=lambda i: i[1]))
    return test_param_balances


def check_sublists_same_size(test_values):
    assert type(test_values) is list, "Test values must be of type list."
    for i in test_values:
        assert type(i) == str or type(i) == list, "Elements of sublist must be list or string"
    list_count = None
    for i in test_values:
        if type(i) == list:
            if list_count is None:
                list_count = len(i)
            else:
                assert len(i) == list_count, f"Sublist {i} length must be the number of sublists {list_count} in the lists. (Excludes str)"
    return list_count


def print_optimisation_outputs(test_param_balances):
    print("Outputs from optimisation:")
    for idx, test in enumerate(test_param_balances):
        print(f"{idx+1}) {test}")
