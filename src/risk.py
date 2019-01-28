import numpy as np

class RiskMetrics:
    def __init__(self):
        self.max_drawdown = []
        self.max_upside = []
        self.returns = []
        self.sharpe_ratio = None
        self.current_trade = self.reset_current()
        
        
    def update_current(self, diff):
        if (self.current_trade['max_drawdown'] == None) or (self.current_trade['max_upside'] == None):
            self.current_trade['max_drawdown'] = diff
            self.current_trade['max_upside'] = diff
        elif diff < self.current_trade['max_drawdown']:
            self.current_trade['max_drawdown'] = diff
        elif diff > self.current_trade['max_upside']:
            self.current_trade['max_upside'] = diff
        
        
    def close_current(self, diff):
        self.returns.append(diff)
        self.max_drawdown.append(self.current_trade['max_drawdown'])
        self.max_upside.append(self.current_trade['max_upside'])
        self.current_trade = self.reset_current()
        
    
    def reset_current(self):
        return {'max_drawdown':None,
                'max_upside':None}
    
    
    def post_analysis(self):
        total = sum(self.returns)
        max_loss = min(self.returns)
        if not max_loss < 0:
            print("No loss occurred, possibly an error in the model or too small a sample")
        max_gain = max(self.returns)
        mean = total/len(self.returns)
        variance = np.var(self.returns)
        sharpe = mean / variance
        return_maxdrawdown = total / max_loss * -1
        upside_over_drawdown = max_gain / max_loss * -1
        
        analysis_results = {"total":total,
                            "trades":len(self.returns),
                            "mean":mean,
                            "max_loss":max_loss,
                            "max_gain":max_gain,
                            "sharpe":sharpe,
                            "RoMDD":return_maxdrawdown,
                            "USoDD":upside_over_drawdown}
        
        return analysis_results
        
    
    
    
        
