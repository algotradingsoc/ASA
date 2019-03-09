import socket_messaging
import time
import csv
import random

PACKING = '?ifi' ## format data is sent over the socket to bokeh

def parallel_backtest(agents_list, send_to_socket=False):
    """Runs multiple against local backtesting file in parallel.
    :param agents_list: list of initalised agents
    :param send_to_socket: Determines if to sent to socket to a bokeh server
    :return: list of the ran agents so perforance data can be extracted from them
    """
    backtest = same_backtest(agents_list)
    if send_to_socket:
        socket_messaging.send_data(PACKING, True, len(agents_list), 0, 0)
    with open(backtest, newline='', encoding='utf-16') as csvfile:
        reader = csv.reader(csvfile)
        try:
            for row_idx,row in enumerate(reader):
                if row_idx % 1000 == 0:
                    print(row_idx)
                for agent_idx, agent in enumerate(agents_list):
                    if row[0] == 'tick':
                        # Check if time column exists
                        tim = datetime.strptime(row.pop(), agent.time_format) if len(row) > 3 else None
                        agent._last_tick = tuple([float(x) for x in row[1:]])
                        agent.on_tick(*agent._last_tick, time=tim)
                    elif row[0] == 'bar':
                        # Check if time column exists
                        tim = datetime.strptime(row.pop(), agent.time_format) if len(row) > 5 else None
                        agent.on_bar(*[float(x) for x in row[1:]], time=tim)
                        if send_to_socket:
                            socket_messages.send_data(PACKING, 
                                                      False, agent_idx, 
                                                      agent.balance, 
                                                      agent.bar_number)
        except KeyboardInterrupt:
            
            ## could add in a way for the socket to send a close message, maybe true, -1, 0, 0?
            pass
    return agents_list

        
def same_backtest(agents_list):
    """Raises an error if the backtest files of initalised agents don't match
    :param agents_list: list of initalised agents
    :return: backtest filename
    """
    backtest=""
    for idx, agent in enumerate(agents_list):
        if idx == 0:
            backtest = agent.backtest
        elif agent.backtest != backtest:
            raise (f"Error, backtest files inconsistent at position {idx}. {agent.backtest} does not match {backtest}")
    return backtest
    
    
def consecutive_backtest(agents_list):
    """Runs the list of initiated agent clases inputed in a consecutive manner.
    :param agent_list: list of initalised agent classes 
    :return: list of the ran agents so perforance data can be extracted from them
    """
    backtest = same_backtest(agents_list)
    ran_agents = []
    for agent in range(agents_list):
        agent.run()
        ran_agents.append(agent)
    return ran_agents
            

def gen_same_agent_list(num_of_agents, agent, **kwargs):
    """Duplicates the same agent numb_of_agents times into a list for backtesting. Returns list
    :param num_of_agents: number of times the agent will be duplicated
    :param agent: agent class to be duplicated, e.g. RandomAgent from rnd_agent.py
    :param kwargs: the rest of the arguments used to initalise the agents
    :return: list of num_of_agent copies of the same initalised agent
    """
    agents = []
    for i in range(num_of_agents):
        agents.append(agent(**kwargs))
    return agents


def gen_random_agent_list(Agent, num_of_agents, 
                          param_names, param_lower, param_upper, 
                          **kwargs):
    """Generates a list of agents with different parameters that is varied randomly between a range
    :param param_names: a list of the two parameter names to be varied eg ['param1', 'param2'] 
    :param param_lower: a list of the two lower bounds 
    :param param_upper: a list of the two upper bounds
    :param num_of_agents: the number of agents to be returned in the list
    :param kwargs: the rest of the arguments used to initalise the agents
    :return: list of num_of_agents with the parameters slightly varied
    """
    agents = []
    assert len(param_names) == 2 ## current set at 2 for ease  
    
    for i in range(num_of_agents):
        param1 = random.randrange(param_lower[0], param_upper[0])
        param2 = random.randrange(param_lower[1], param_upper[1])
        params = {param_names[0] : param1, param_names[1] : param2}
        agent_inputs = {**params, **kwargs}
        agents.append(Agent(**agent_inputs))
    return agents

        
def print_results_summary(ran_agents):
    """Displays some results from a list of agents that have ran
    :return: list of results from the agents
    """
    results = []
    total_return = 0
    for agent in ran_agents:
        results = agent.risk.post_analysis()
        total_return += results['total']
        print("total:", results['total'])
        print("# of trades:",results['trades'])
        print("return/max drawdown:", results['RoMDD'])
        print("return:", agent.balance)
        print("============")
    print("mean return:", total_return/len(ran_agents))
    print("============")
    
    
def rnd_agent_example(filename, num_of_agents):
    from agent_rnd import RandomAgent
    agents_list = gen_same_agent_list(num_of_agents, RandomAgent, 
                                      choice_on_tick=True, 
                                      backtest=filename)
    ran_agents = parallel_backtest(agents_list, send_to_socket=False)
    results = print_results_summary(ran_agents)
    
    
def mac_agent_example(filename, num_of_agents):
    from agent_moving_avg_cross import MACAgent
    agent_list = gen_random_agent_list(MACAgent, num_of_agents,
                                       ['ma_1_length', 'ma_2_length'],
                                       [5,50],
                                       [1000,2000],
                                       backtest=filename)
    ran_agents = parallel_backtest(agent_list, send_to_socket=False)
    print_results_summary(ran_agents)
    
    results = []
    for agent in ran_agents:
        results.append([agent.fast_period, agent.slow_period, agent.balance])
    results.sort(key=lambda x: x[2])
    for result in results:
        print(f"{result[0]} \t {result[1]} \t {result[2]}")
           
            
if __name__=='__main__':
    num_of_agents = 200
    filename = "data/1yr_backtest_GBPUSD.csv"
    
    from backtest_funcs import get_file_length
    length = get_file_length(filename)
    print(f"Backtest file length: {length}")
    
    mac_agent_example(filename, num_of_agents)
        
    