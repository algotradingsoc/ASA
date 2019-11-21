from pedlar.agent import Agent
import csv

class BuildDatasetAgent(Agent):
    '''A trading agent.'''
    
    def __init__(self, file_name, **kwargs):
        super().__init__(**kwargs)
        self.file_name = file_name

    def on_tick(self, bid, ask, time=None):
        '''Called on every tick update.'''
        time_string = time.strftime('%Y.%m.%d %H:%M:%S')
        print('Tick:', bid, ask, time_string)
        with open(self.file_name, 'a', 
                  newline='', encoding='utf-16') as fd:
            writer = csv.writer(fd, delimiter=',')
            writer.writerow(['tick', bid, ask, time_string])

if __name__ == '__main__':
    agent = BuildDatasetAgent(file_name='data/GBPUSD_161119.csv',
                              username='joe', password='1234',
                              ticker='tcp://icats.doc.ic.ac.uk:7000',
                              endpoint='http://icats.doc.ic.ac.uk')
    agent.run()
