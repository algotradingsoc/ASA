from pedlar.agent import Agent

class EchoAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def on_tick(self, bid, ask, time=None):
        print(f'{time} : {bid} {ask}')
        
        
if __name__=='__main__':
    backtest = True
    if backtest:
        agent = EchoAgent(backtest='data/GBPUSD_161119.csv')
    else:
        agent = EchoAgent(username='joe', password='1234',
                          ticker='tcp://icats.doc.ic.ac.uk:7000',
                          endpoint='http://icats.doc.ic.ac.uk')
    agent.run()









