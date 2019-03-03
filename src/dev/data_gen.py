import random
import matplotlib.pyplot as plt

class Brown():
    def __init__(self, start=0, var=5, record=False):
        self.x = start
        self.var = var
        self.record = record
        if self.record:
            self.hist = [self.x]
        
    def add(self, scale):
        move = random.randint(-self.var,self.var)
        if scale:
            self.x += (move / (self.var * 100)) * self.x
        else:
            self.x += move
        if self.record:
            self.hist.append(self.x)
                
    def run_add(self, steps=100, scale=False):
        for i in range(steps):
            self.add(scale)    

    def visual(self):
        assert self.record ## need to initalise with record=True
        plt.plot(self.hist)
        plt.show()

def log_return_visual(log_returns):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,12))
    for c in log_returns:
        ax1.plot(log_returns.index, log_returns[c].cumsum(), label=str(c))
        
    ax1.set_ylabel('Cum log returns')
    ax1.legend(loc='best')
    
    for c in log_returns:
        ax2.plot(log_returns.index, 100*(np.exp(log_returns[c].cumsum()) - 1), label=str(c))
        
    ax2.set_ylabel('Total relative returns (%)')
    ax2.legend(loc='best')
    
    plt.show()
    

def get_returns(log_returns):
    ## last day returns
    r_t = log_returns.tail(1).transpose()

    weights_vector = pd.DataFrame(1/3, index=r_t.index, columns=r_t.columns)
    
    portfolio_log_returns = weights_vector.transpose().dot(r_t)
    return portfolio_log_returns
    
def get_portfolio_return(log_returns):
    weights_matrix = pd.DataFrame(1/3, index=log_returns.index, columns=log_returns.columns)
    
    temp_var = weights_matrix.dot(log_returns.transpose())
    #print(temp_var.head().iloc[:, 0:5])
    
    portfolio_log_returns = pd.Series(np.diag(temp_var), index=log_returns.index)
    #print(portfolio_log_returns.tail()) 
    
    total_relative_returns = (np.exp(portfolio_log_returns.cumsum()) - 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,12))
    
    ax1.plot(portfolio_log_returns.index, portfolio_log_returns.cumsum())
    ax1.set_ylabel('Portfolio cum log returns')
    
    ax2.plot(total_relative_returns.index, 100 * total_relative_returns)
    ax2.set_ylabel('Portfolio total relative returns (%)')
    
    plt.show()
    
    days_per_year = 52 * 5
    total_days_in_simulation = log_returns.shape[0]
    number_of_years = total_days_in_simulation / days_per_year
    
    total_portfolio_return = total_relative_returns.iloc[-1]

    average_yearly_return = (1 + total_portfolio_return)**(1 / number_of_years) - 1
    
    print(f"Total portfolio return: {total_portfolio_return * 100}")
    print(f"Average yearly return: {average_yearly_return * 100}")
        
        
def sma_visual(df, short_period=20, long_period=200):
    short_rolling = df.rolling(window=short_period).mean()
    long_rolling = df.rolling(window=long_period).mean()
    
    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(df.index, df.loc[:,'0'], label='Price')
    ax.plot(long_rolling.index, long_rolling.loc[:,'0'], label = "200 day SMA")
    ax.plot(short_rolling.index, short_rolling.loc[:,'0'], label = "20 day SMA")
    
    ax.legend(loc='best')
    ax.set_ylabel('Price in $')
    
    plt.show()
        

def ema_visual(df, period=20):
    ema_short = df.ewm(span=period, adjust=False).mean()
    
    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(df.index, df.loc[:,:], label='Price')
    ax.plot(ema_short.index, ema_short.loc[:,:], label = "250 day EMA")
    
    ax.legend(loc='best')
    ax.set_ylabel('Price in $')
    
    plt.show()
    
def ema_strat(df, period=20, visual=None):
    ema_short = df.ewm(span=period, adjust=False).mean()
    trading_positions_raw = df - ema_short
    trading_positions = trading_positions_raw.apply(np.sign) * 1/3
    trading_positions_final = trading_positions.shift(1)
    if visual is not None:
        visual_ema_strat(df, ema_short, trading_positions_final, visual)
        
    asset_log_returns = np.log(df).diff()
    print(asset_log_returns.cumsum().tail(1)['0'])
    
        
    
def visual_ema_strat(df, ema_short, trading_positions_final, asset):
    asset_log_returns = np.log(df).diff()
    print(asset_log_returns.cumsum().tail(1)['0'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16,9))
    
    ax1.plot(df.index, df.loc[:,asset], label=f'Price ')
    ax1.plot(ema_short.index, ema_short.loc[:,asset], label=f'EMA')
    
    ax1.set_ylabel('$')
    ax1.legend(loc='best')
    
    ax2.plot(trading_positions_final.index, trading_positions_final.loc[:,asset], label='Trading position')
    ax2.set_ylabel('Trading position')
    
    ax3.plot(asset_log_returns.index, asset_log_returns.cumsum().loc[:,asset], label="Cumulative Log Returns")
    ax3.set_ylabel('Trading position')
    
    plt.show()
    
        
if __name__=='__main__':
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame()
    for i in range(3):
        x1 = Brown(start=1000, record=True)
        x1.run_add(2000, scale=True)
        df[f'{i}'] = pd.Series(x1.hist)
    
    ## relative returns 
    #returns = df.pct_change(1)
    #log_returns = np.log(df).diff()
    
    #log_return_visual(log_returns)
    #print(get_portfolio_return(log_returns))
    #sma_visual(df)
    
    ema_strat(df, period=500, visual='0')
    
    
    