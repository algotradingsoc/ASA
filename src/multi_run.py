import socket_messaging
import time
import csv

        
def parallel_backtest(agents_list, send_to_socket=False):
    """Runs multiple against local backtesting file in parallel.
    :param agents_list: list of initalised agents
    :param send_to_socket: Determines if to sent to socket to a bokeh server
    :return: list of the ran agents so perforance data can be extracted from them
    """
    backtest = same_backtest(agents_list)
    if send_to_socket:
        socket_messaging.init_bokeh(len(agents_list))
    with open(backtest, newline='', encoding='utf-16') as csvfile:
        reader = csv.reader(csvfile)
        try:
            for row_idx,row in enumerate(reader):
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
                            socket_messaging.send_agent_data(agent_idx, agent.balance, agent.bar_number)
        except KeyboardInterrupt:
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
        
        
def print_results_summary(ran_agents):
    """Displays some results from a list of agents that have ran
    :return: list of results from the agents
    """
    results = []
    for agent in ran_agents:
        results = agent.risk.post_analysis()
        print(results['total'])
        print(results['trades'])
        print(results['RoMDD'])
        print('============')
    return results
        
                   
if __name__=='__main__':
    num_of_agents = 12
    backtest = "data/1yr_backtest_GBPUSD.csv"
    choice_on_tick=True
    send_to_socket=True
    from agent_rnd import RandomAgent
    agents_list = gen_same_agent_list(num_of_agents, RandomAgent, choice_on_tick=choice_on_tick, backtest=backtest)
    ran_agents = parallel_backtest(agents_list, send_to_socket=send_to_socket)
    results = print_results_summary(ran_agents)
    