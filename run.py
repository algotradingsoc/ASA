
def get_kwargs_from_args(args):
    try:
        kwargs = {}
        for i in args.agent_params:
            key, value = i.split("=")
            if key == "verbose":
                raise Exception("agent_params can not have a key of verbose")
            elif key == "backtest":
                raise Exception("agent_params can not have a key of backtest")
            kwargs[key] = float(value)
    except ValueError:
        raise ValueError("agent_params expected to be inputted in the form: 'argument=value'")
    return kwargs


def run_agent_from_args(args):
    kwargs = get_kwargs_from_args(args)
    backtest_path = args.backtest_path if args.backtest_path != "None" else None

    if args.agent == 0:
        from Agents.agent_0_echo import main as agent_0_main
        agent_0_main(backtest=backtest_path)
    elif args.agent == 1:
        from Agents.agent_1_simple_macd import main as agent_1_main
        agent_1_main(signal_mean_length=args.signal_mean_length,
                     make_orders=args.no_make_order,
                     verbose=args.verbose,
                     backtest=backtest_path, **kwargs)
    elif args.agent == 2:
        from Agents.agent_2_simple_risk_managed_macd import main as agent_2_main
        agent_2_main(signal_mean_length=args.signal_mean_length,
                     make_orders=args.no_make_order,
                     verbose=args.verbose,
                     backtest=backtest_path, **kwargs)
    elif args.agent == 3:
        from Agents.agent_3_ret_bound_risk_macd import main as agent_3_main
        agent_3_main(signal_mean_length=args.signal_mean_length,
                     make_orders=args.no_make_order,
                     verbose=args.verbose,
                     backtest=backtest_path, **kwargs)
    elif args.agent == 4:
        from Agents.agent_4_decision_tree import main as agent_4_main
        agent_4_main(signal_mean_length=args.signal_mean_length,
                     make_orders=args.no_make_order,
                     verbose=args.verbose,
                     backtest=backtest_path, **kwargs)
    elif args.agent == 5:
        raise NotImplementedError
    elif args.agent == 6:
        from Agents.agent_6_linear_model import main as agent_6_main
        agent_6_main(rets_length=args.rets_length,
                     signal_mean_length=args.signal_mean_length,
                     make_orders=args.no_make_order,
                     verbose=args.verbose,
                     backtest=backtest_path, **kwargs)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    backtest_default = 'data/backtest_GBPUSD_12_hours.csv'
    import argparse
    parser = argparse.ArgumentParser(description='Parameters to determine which agent should run and with what parameters.')
    parser.add_argument('--backtest_path', type=str, default=backtest_default, help="Path of backtest file, enter None for live data")
    parser.add_argument('--no_make_order', action='store_false', help="Makes the agent only estimate orders")
    parser.add_argument('--rets_length', type=int, default=5, help="Length of the rets deque")
    parser.add_argument('--signal_mean_length', type=int, default=1, help="Length of the mean from signals")
    parser.add_argument('--verbose', action='store_true', help="Enter to make agent verbose")
    parser.add_argument('--agent', type=int, default=0, help="Enter the number of the agent you wish to run, zero by default")
    parser.add_argument('agent_params', nargs='*', help="Enter the parameters for the agent.")
    args = parser.parse_args()

    run_agent_from_args(args)
