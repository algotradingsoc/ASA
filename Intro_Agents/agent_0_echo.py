from pedlar.agent import Agent


class EchoAgent(Agent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # initialise the parent Agent class

    def on_tick(self, bid, ask, time=None):
        ''' Every time the agent recieves a tick:
                - This function is called.
                - The function prints out the current tick
        '''
        print(f'Tick: {bid: .05f}, {ask: .05f}, {time}')


if __name__ == '__main__':
    backtest = True
    if backtest:
        # Uses the csv file to backtest the agent.
        agent = EchoAgent(backtest='../data/backtest_GBPUSD_12_hours.csv')
    else:
        # Receives ticks from the pedlar server using credentials.
        agent = EchoAgent(username='joe', password='1234',
                          ticker='tcp://icats.doc.ic.ac.uk:7000',
                          endpoint='http://icats.doc.ic.ac.uk')
    agent.run()
