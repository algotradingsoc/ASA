class AgentCore():
    def __init__(self):
        self.diff = None
        self.bid = None
        self.ask = None
        self.mid = None
        self.spread = None
    
    
    def update_bid_ask_mid_spread(self, bid, ask):
        self.bid, self.ask = bid, ask 
        self.mid = (ask + bid)/2
        self.spread = ask - bid
        return
    
    
    def update_diff(self, order):
        if order.type =="buy":
            self.diff = self.bid - order.price
        else:
            self.diff = order.price - self.ask
        return
    
    def get_file_length(self, filename):
        with open(filename, newline='', encoding='utf-16') as csvfile:
                reader = csv.reader(csvfile)
                length = sum(1 for row in reader)
        return length
    